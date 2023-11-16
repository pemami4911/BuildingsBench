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
from buildings_bench.models.clip import CLIP
from buildings_bench.evaluation.metrics import MetricType
from buildings_bench.evaluation import metrics_factory
from buildings_bench.evaluation import scoring_rule_factory
from buildings_bench.data import g_weather_features

SCRIPT_PATH = Path(os.path.realpath(__file__)).parent

def main(args, model_args):
    """ Main training loop
    
    Args:
        args (argparse.Namespace): Command line arguments
        model_args (dict): Model arguments
    """

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


    if args.rank == 0:
        if not checkpoint_dir.exists():
            os.makedirs(checkpoint_dir)
        
        wandb_project = os.environ.get('WANDB_PROJECT', '')
        if wandb_project == '':
            print('WANDB_PROJECT environment variable not set, disabling wandb')
            args.disable_wandb = True
        
        if args.disable_wandb:
            run = wandb.init(
                project=wandb_project,
                mode="disabled",
                config=args)
        elif args.resume_from_checkpoint != '':
            run = wandb.init(
                id=args.wandb_run_id,
                project=wandb_project,
                notes=args.note,
                resume="allow",
                config=args)
        else:
            run = wandb.init(
                project=wandb_project,
                notes=args.note,
                config=args)
    
    global_batch_size = args.world_size * args.batch_size

    #################### Model setup ####################
    if args.weather is not None and args.weather[0] == 'all':
        args.weather = g_weather_features

    # model, loss, predict = model_factory(args.config, model_args)
    model = CLIP(**model_args)
    loss = model.loss
    model = model.to(local_rank)
    print(f'rank {args.rank} number of trainable parameters is '\
          f'= {sum(p.numel() for p in model.parameters())}', flush=True)

    #################### Dataset setup ####################
    train_dataset = load_pretraining("simcap",
                                     args.num_buildings,
                                     args.apply_scaler_transform,
                                     transform_path, 
                                     weather=args.weather, 
                                     custom_idx_filename=args.train_idx_filename,
                                     context_len=0,
                                     pred_len=model.pred_len,
                                     use_buildings_chars=args.use_buildings_chars,
                                     use_text_embedding=args.use_text_embedding,
                                     building_description=True)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                                     dataset=train_dataset,
                                     num_replicas=args.world_size,
                                     rank=args.rank, shuffle=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        drop_last=False, worker_init_fn=utils.worker_init_fn_eulp,
        shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True)
    
    if not model.continuous_loads:
        load_transform = LoadQuantizer(
            with_merge=(not args.tokenizer_without_merge),
            num_centroids=model.vocab_size,
            device=f'cuda:{local_rank}')
        load_transform.load(transform_path)
    # else:
    #     load_transform = train_dataset.load_transform
    elif args.apply_scaler_transform != '':
        load_transform = train_dataset.load_transform
    else:
        load_transform = None

    if not model.continuous_loads: 
        transform = load_transform.transform
        inverse_transform = load_transform.undo_transform
    elif args.apply_scaler_transform != '':
        transform = lambda x: x
        inverse_transform = load_transform.undo_transform
    else: # Continuous unscaled values
        transform = lambda x: x
        inverse_transform = lambda x: x

    #################### Optimizer setup ##########################

    # wrap model with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    print(f'rank {args.rank} wrapped model in DDP', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    train_steps = args.train_tokens // (global_batch_size * model.module.pred_len)

    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                            num_warmup_steps=args.warmup_steps,
                            num_training_steps=train_steps)
        
    scaler = torch.cuda.amp.GradScaler()

    #################### Resume from checkpoint ####################

    if args.resume_from_checkpoint != '':
        model, optimizer, scheduler, step = utils.load_model_checkpoint(
            checkpoint_dir / args.resume_from_checkpoint, model, optimizer, scheduler, local_rank)
        seen_tokens = step * global_batch_size  * model.module.pred_len
    else:
        step = 0
        seen_tokens = 0

    #################### Training loop ##############################
    print(f'rank {args.rank} step {step} train_steps = {train_steps}', flush=True)

    # fix sampling seed such that each gpu gets different part of dataset        
    train_sampler.set_epoch(0)
    model.train()
    start_time = timer()

    while step < train_steps:
        for batch in train_dataloader:
            start_time = timer()
            optimizer.zero_grad()

            for k,v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(model.device)
            
            # Apply transform to load if needed
            batch['load'] = transform(batch['load'])
            
            # backwards is called in here
            with torch.cuda.amp.autocast():
                text_embeddings, physics_embeddings = model(batch) 
                batch_loss = loss(text_embeddings, physics_embeddings)
            
            # Scale Gradients
            scaler.scale(batch_loss).backward()
            
            # Update Optimizer
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            end_time = timer()

            secs_per_step = end_time - start_time
            # world_size * batch_size = global batch size with DDP training
            seen_tokens += (global_batch_size * model.module.pred_len)
            step += 1

            ppl = torch.exp(batch_loss.detach())

            if args.rank == 0 and step % 500 == 0:
                wandb.log({
                    'train/loss': batch_loss,
                    'train/batch_ppl': ppl,
                    'train/seen_tokens (M)': seen_tokens / 1000000,
                    'train/secs_per_step': secs_per_step,
                    'train/lr': optimizer.param_groups[0]['lr']
                }, step=step)

            if step % 10000 == 0:
                if args.note != '':
                    model_name = f'ckpt-step-{step}-{args.note}.pt'
                else:
                    model_name = f'ckpt-step-{step}.pt'
                utils.save_model_checkpoint(model, optimizer, scheduler, step, checkpoint_dir / model_name)
        
            if step >= train_steps:
                # stop training after this many steps/train_tokens
                break
        else:
            continue # continue if step < train_steps
        break

    # save final checkpoint
    if args.rank == 0:
        if args.note != '':
            model_name = f'ckpt-step-{step}-{args.note}.pt'
        else:
            model_name = f'ckpt-step-{step}.pt'
        utils.save_model_checkpoint(model, optimizer, scheduler, step, checkpoint_dir / model_name)

    torch.distributed.destroy_process_group()        
    run.finish()


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
    parser.add_argument('--resume_from_checkpoint', type=str, default='')
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
    parser.add_argument('--train_idx_filename', type=str, default='',
                        help='Name of index files for training')
    parser.add_argument('--val_idx_filename', type=str, default='',
                        help='Name of index files for validation')
    parser.add_argument('--use-weather', dest='weather', nargs='*', default=None,
                    help='Enable loading weather features (they are not used by default). If enabled, all EULP\'s weather features will be loaded. '
                    'Optionally, specify a list of weather features to use (see `weather_features` in buildings_bench.data.__init__.py for options')
    parser.add_argument('--use_buildings_chars', default=True, help="Whether include building chars for surrogate training, default = True.")
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
    main(experiment_args, model_args)