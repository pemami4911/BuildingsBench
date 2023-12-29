import pandas as pd
from pathlib import Path
import os
from buildings_bench.data.buildings900K import Buildings900K
from collections import defaultdict
import numpy as np

metadata_path = Path("/projects/foundation/eulp/v1.1.0/BuildingsBench/metadata_dev")
g_weather_features = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 'global_horizontal_radiation', 
                              'direct_normal_radiation', 'diffuse_horizontal_radiation']
dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))
dataset = Buildings900K(dataset_path,
                       index_file=metadata_path / "comcap_350k_tune.idx",
                       context_len=0,
                       pred_len=1,
                       weather=g_weather_features,
                       onehot_encode=False,
                       use_buildings_chars=True,
                       use_text_embedding=False,
                       building_description=False,
                       surrogate_mode=True)

data = defaultdict(list)
features = ["latitude", "longitude", "day_of_year", "day_of_week", "hour_of_day", \
                    "load", "building_char", "building_id", "dataset_id"] + g_weather_features
for i in range(len(dataset)):
    if i % 1000 == 0:
        print(i)
    building_data = dataset[i]
    for feature in features:
        if feature in ["building_id", "dataset_id"]:
            data[feature].append(np.repeat(np.array([[building_data[feature]]]), dataset.pred_len, axis=0))
        else:
            data[feature].append(building_data[feature])

for feature in features:
    data[feature] = np.vstack(data[feature])

for feature in features:
    print(data[feature].shape)

X_feature = ["latitude", "longitude", "day_of_year", "day_of_week", "hour_of_day"] + g_weather_features + ["building_char"]
X = np.hstack([data[f] for f in X_feature])
np.savez(metadata_path / "comcap_X_day.npz", X=X)

Y = data["load"]
np.savez(metadata_path / "comcap_Y_day.npz", Y=Y)

meta = np.hstack([data[f] for f in ["building_id", "dataset_id"]])
np.savez(metadata_path / "comcap_meta_day.npz", meta=meta)

