import torch
import torch.nn as nn
from pathlib import Path
import transformers
import argparse 
import wandb
import os
import tomli
from timeit import default_timer as timer
from socket import gethostname
from buildings_bench import utils
from buildings_bench import BuildingTypes
from buildings_bench import load_pretraining
from buildings_bench.tokenizer import LoadQuantizer
from buildings_bench.evaluation.managers import MetricsManager
from buildings_bench.models import model_factory
from buildings_bench.evaluation.metrics import MetricType
from buildings_bench.evaluation import metrics_factory
from buildings_bench.evaluation import scoring_rule_factory
from buildings_bench.data import load_torch_dataset
from buildings_bench.data import g_weather_features
import numpy as np
import math
import pickle

class statsTracker:
    def __init__(self):
        self.counter = None
        self.average = None
    
    def update(self, x):
        if self.counter is None:
            self.counter = x.size(0)
            self.average = x.mean(dim=0)
        else:
            self.average = x.sum(dim=0) + self.average * self.counter
            self.counter += x.size(0)
            self.average = self.average / self.counter

    def get(self):
        return self.average

class uniqueTracker:
    def __init__(self):
        self.unique = set()

    def update(self, x):
        for row in x:
            self.unique.add((row[0], row[1]))

    def get(self):
        return len(self.unique)

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent

@torch.no_grad()
def aggregate_eval(args, model_args, testset="val"):
    utils.set_seed(args.random_seed)

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # Optimize for fixed input sizes
    torch.backends.cudnn.benchmark = False

    ######################### DDP setup  #########################
    # SLURM_LOCALID: gpu local rank (=0 as the first gpu of the node)
    # SLURM_PROCID: gpu global rank (=4 as the fifth gpu among the 8)
    # MASTER_ADDR and MASTER_PORT env variables should be set when calling this script
    gpus_per_node = torch.cuda.device_count()    
    args.world_size    = int(os.environ["WORLD_SIZE"])
    if args.disable_slurm:
        local_rank     = int(os.environ["LOCAL_RANK"])
        args.rank      = local_rank
    else:
        args.rank      = int(os.environ["SLURM_PROCID"])
        print(f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are" \
            f" {gpus_per_node} allocated GPUs per node.", flush=True)

        local_rank = args.rank - gpus_per_node * (args.rank // gpus_per_node)

    print(f'About to call init_process_group on rank {args.rank} with local rank {local_rank}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, 
                                        init_method=args.dist_url,
                                        world_size=args.world_size,
                                        rank=args.rank)
    if args.rank == 0: print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)
    torch.cuda.set_device(local_rank)

    print(f'rank {args.rank} torch cuda available = ', torch.cuda.is_available(), flush=True)
    print(f'rank {args.rank} torch cuda device count = ', torch.cuda.device_count(), flush=True)
    print(f'rank {args.rank} torch cuda current device = ', torch.cuda.current_device(), flush=True)
    print(f'rank {args.rank} torch cuda get_device_name = ', torch.cuda.get_device_name(0), flush=True)
    print(f'rank {args.rank} torch threads = ', torch.get_num_threads(), flush=True)

    print(f'dataset path = {os.environ.get("BUILDINGS_BENCH", "")}')

    checkpoint_dir = SCRIPT_PATH / '..' / 'checkpoints'
    transform_path = Path(os.environ.get('BUILDINGS_BENCH', '')) / 'metadata' / 'transforms'

    #################### Model setup ####################

    # remove "module" from state_dict keys to load weights without data parallel
    # source: https://gist.github.com/IAmSuyogJadhav/bc388a871eda982ee0cf781b82227283
    # from collections import OrderedDict
    # def remove_data_parallel(old_state_dict):
    #     new_state_dict = OrderedDict()

    #     for k, v in old_state_dict.items():
    #         name = k[7:] # remove `module.`
    #         new_state_dict[name] = v
    
    #     return new_state_dict
    if args.weather is not None and args.weather[0] == 'all':
        args.weather = g_weather_features

    model, loss, predict = model_factory(args.config, model_args)
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    model.load_state_dict(
        torch.load(f"checkpoints/{args.checkpoint}", map_location=f'cuda:{local_rank}')["model"]
    )
    
    print(f'rank {args.rank} number of trainable parameters is '\
          f'= {sum(p.numel() for p in model.parameters())}', flush=True)

    #################### Dataset setup ####################

    building_dataset = load_pretraining('simcap',
                            args.num_buildings,
                            args.apply_scaler_transform,
                            transform_path, 
                            weather=args.weather, 
                            context_len=0,
                            pred_len=24,
                            custom_idx_filename=args.test_idx_filename,
                            use_buildings_chars=True,
                            use_text_embedding=args.use_text_embedding
                        )
    
    if not model.module.continuous_loads:
        load_transform = LoadQuantizer(
            with_merge=(not args.tokenizer_without_merge),
            num_centroids=model.module.vocab_size,
            device=f'cuda:{local_rank}')
        load_transform.load(transform_path)
    # else:
    #     load_transform = train_dataset.load_transform
    elif args.apply_scaler_transform != '':
        load_transform = building_dataset.load_transform
    else:
        load_transform = None

    if not model.module.continuous_loads: 
        transform = load_transform.transform
        inverse_transform = load_transform.undo_transform
    elif args.apply_scaler_transform != '':
        transform = lambda x: x
        inverse_transform = load_transform.undo_transform
    else: # Continuous unscaled values
        transform = lambda x: x
        inverse_transform = lambda x: x

    model.eval()
    step = 0

    bd_types = ['FullServiceRestaurant', 'Hospital', 'LargeHotel', 'LargeOffice',
       'MediumOffice', 'Outpatient', 'PrimarySchool',
       'QuickServiceRestaurant', 'RetailStandalone', 'RetailStripmall',
       'SecondarySchool', 'SmallHotel', 'SmallOffice', 'Warehouse']

    stats = [[statsTracker(), statsTracker(), uniqueTracker()] for _ in range(14)]

    test_sampler = torch.utils.data.distributed.DistributedSampler(
                                    dataset=building_dataset,
                                    num_replicas=args.world_size,
                                    rank=args.rank, shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(
        building_dataset, batch_size=args.batch_size, sampler=test_sampler,
        drop_last=False, worker_init_fn=utils.worker_init_fn_eulp if testset == "val" else None,
        shuffle=(test_sampler is None), num_workers=args.num_workers, pin_memory=True)

    for batch in test_dataloader:   
        # building_types_mask = batch['building_type'][:,0,0] == 1
        building_types_mask = batch['building_subtype'] != -1
        building_types_mask = building_types_mask.cpu()

        for k,v in batch.items():
            batch[k] = v.to(model.device)

        continuous_targets = batch['load'].clone()

        # Transform if needed
        batch['load'] = transform(batch['load'])
        targets = batch['load']

        with torch.cuda.amp.autocast():
            # preds = model(batch)
            # batch_loss = loss(preds, targets)
            predictions, distribution_params = predict(batch)

        predictions = inverse_transform(predictions)

        if args.apply_scaler_transform != '':
            continuous_targets = inverse_transform(continuous_targets)
            # unscale for crps
            targets = inverse_transform(targets)
            if args.apply_scaler_transform == 'standard' and distribution_params is not None:
                mu = inverse_transform(distribution_params[:,:,0])
                sigma = load_transform.undo_transform_std(distribution_params[:,:,1])
                distribution_params = torch.cat([mu.unsqueeze(-1), sigma.unsqueeze(-1)],-1)
            
            elif args.apply_scaler_transform == 'boxcox' and distribution_params is not None:
                ######## approximate Gaussian in unscaled space ########
                mu = inverse_transform(distribution_params[:,:,0])
                muplussigma = inverse_transform(torch.sum(distribution_params,-1))
                sigma = muplussigma - mu
                muminussigma = inverse_transform(distribution_params[:,:,0] - distribution_params[:,:,1])
                sigma = (sigma + (mu - muminussigma)) / 2
                distribution_params = torch.cat([mu.unsqueeze(-1), sigma.unsqueeze(-1)],-1)
        
        if not model.module.continuous_loads:
            centroids = load_transform.kmeans.centroids.squeeze() \
                if args.tokenizer_without_merge else load_transform.merged_centroids
        else:
            centroids = None
                    
        step += 1

        pred = predictions.cpu()[:, :, 0]
        targ = targets.cpu()[:, :, 0]
        time = batch["hour_of_day"].cpu()[:, :, 0]
        bd_ids = torch.cat((batch["dataset_id"].unsqueeze(1), batch["building_id"].unsqueeze(1)), dim=1).long()

        for i in range(14):
            mask = batch['building_subtype'] == i
            mask = mask.cpu()
            if mask.any():
                idx = time[mask, :].argmin(dim=1)

                p = pred[mask, :].repeat(1, 2)
                t = targ[mask, :].repeat(1, 2)
                p = torch.cat([p[j, id:id + 24].unsqueeze(0) for j, id in enumerate(idx)])
                t = torch.cat([t[j, id:id + 24].unsqueeze(0) for j, id in enumerate(idx)])
                
                stats[i][0].update(p.double())
                stats[i][1].update(t.double())
                stats[i][2].update(bd_ids[mask, :])

    with open(f'{args.note}.pickle', 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
    for i in range(14):
        plt.subplot(4, 4, i+1)
        p = stats[i][0].get()
        t = stats[i][1].get()
        n = stats[i][2].get()

        print(bd_types[i])
        print(p, t, n)

        plt.plot(p, label="prediction")
        plt.plot(t, label="target")
        plt.title(bd_types[i])
        plt.legend()
        plt.tight_layout()

    plt.savefig(f'{args.note}-plot.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Experiment args. If provided in config file, these will be overridden.
    # Use arg `hyper_opt` to avoid overriding the argparse args with the config file.
    parser.add_argument('--config', type=str, default='', required=True,
                        help='Name of your model. Should match the config'
                             ' filename without .toml extension.'
                             ' Example: "TransformerWithTokenizer-S"')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.00006)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--train_tokens', type=int, default=1000000000) # 1B
    parser.add_argument('--random_seed', type=int, default=99)
    parser.add_argument('--ignore_scoring_rules', action='store_true',
                        help='Do not compute a scoring rule for this model.')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--hyper_opt', nargs='*', default=[],
                        help='Tells this script to not override the argparse values for'
                             ' these hyperparams with values in the config file.'
                             ' Expects the hyperparameter value to be set via argparse '
                             ' from the CLI. Example: --hyper_opt batch_size lr')
    
    # Wandb
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--note', type=str, default='',
                        help='Note to append to model checkpoint name. '
                        'Also used for wandb notes.')    
    parser.add_argument('--wandb_run_id', type=str, default='')

    # DDP
    parser.add_argument('--disable_slurm', action='store_true')
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--num_workers', type=int, default=8)

    # Variants
    parser.add_argument('--num_buildings', type=int, default=-1,
                        help='Number of buildings to use for training. '
                             'Default is -1 which uses all buildings. ' 
                             'Options {1000, 10000, 100000}.')
    parser.add_argument('--tokenizer_without_merge', action='store_true', default=False, 
                        help='Use the tokenizer without merge. Default is False.')
    parser.add_argument('--apply_scaler_transform', type=str, default='',
                        choices=['', 'standard', 'boxcox'], 
                        help='Apply a scaler transform to the load values.')
    parser.add_argument('--test_idx_filename', type=str, default='',
                        help='Name of index files for testing')
    parser.add_argument('--use-weather', dest='weather', nargs='*', default=None,
                    help='Enable loading weather features (they are not used by default). If enabled, all EULP\'s weather features will be loaded. '
                    'Optionally, specify a list of weather features to use (see `weather_features` in buildings_bench.data.__init__.py for options')
    parser.add_argument('--use_text_embedding', action='store_true', default=False, 
                    help="Whether to use text embeddings of building descriptions. If false, use one-hot encoded features instead. Default is False.")

    experiment_args = parser.parse_args()

    if experiment_args.weather == []:
        experiment_args.weather = ['all']

    # validate hyperopt args, if any
    for arg in experiment_args.hyper_opt:
        if not hasattr(experiment_args, arg):
            raise ValueError(f'Hyperopt arg {arg} not found in argparse args.')
        
    config_path = SCRIPT_PATH  / '..' / 'buildings_bench' / 'configs'
    
    if (config_path / f'{experiment_args.config}.toml').exists():
        toml_args = tomli.load(( config_path / f'{experiment_args.config}.toml').open('rb'))
        model_args = toml_args['model']
        if 'pretrain' in toml_args:
            for k,v in toml_args['pretrain'].items():
                if not k in experiment_args.hyper_opt:
                    if hasattr(experiment_args, k):
                        print(f'Overriding argparse default for {k} with {v}')
                    # Just set the argparse value to the value in the config file
                    # even if there is no default
                    setattr(experiment_args, k, v)
        if not model_args['continuous_loads'] or 'apply_scaler_transform' not in experiment_args:
            setattr(experiment_args, 'apply_scaler_transform', '')
    else:
        raise ValueError(f'Config {experiment_args.config}.toml not found.')

    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available for pretraining!')
    
    print("start training")
    aggregate_eval(experiment_args, model_args)