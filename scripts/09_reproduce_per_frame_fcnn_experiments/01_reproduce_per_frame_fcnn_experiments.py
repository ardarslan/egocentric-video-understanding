import os

os.environ["CODE"] = "/home/aarslan/mq"
os.environ["SLURM_CONF"] = "/home/sladmcvl/slurm/slurm.conf"
os.environ["SCRATCH"] = "/srv/beegfs-benderdata/scratch/aarslan_data/data"
os.environ["CUDA_HOME"] = "/usr/lib/nvidia-cuda-toolkit"

import sys

sys.path.append("../08_reproduce_mq_experiments/")

from libs.core import load_config
from libs.utils import fix_random_seed
from libs.datasets import make_dataset, make_data_loader

config = "/home/aarslan/mq/scripts/08_reproduce_mq_experiments/configs/proposed_features_v6.yaml"

cfg = load_config(config)
cfg["loader"]["num_workers"] = 0
cfg["dataset_name"] = "ego4d_per_frame"

for i in range(len(cfg["dataset"]["video_feat_folder"])):
    cfg["dataset"]["video_feat_folder"][i] = os.path.join(
        os.environ["SCRATCH"], cfg["dataset"]["video_feat_folder"][i]
    )

rng_generator = fix_random_seed(cfg["init_rand_seed"], include_cuda=True)

train_dataset = make_dataset(
    cfg["dataset_name"], True, cfg["train_split"], **cfg["dataset"]
)
train_loader = make_data_loader(train_dataset, True, rng_generator, **cfg["loader"])

val_dataset = make_dataset(
    cfg["dataset_name"], False, cfg["val_split"], **cfg["dataset"]
)
# set bs = 1, and disable shuffle
val_loader = make_data_loader(val_dataset, False, None, 1, cfg["loader"]["num_workers"])

for batch in train_loader:
    print(batch["feats"].shape, batch["segmentation_labels"].shape)
    break

    # a = 2
    # break

# for batch in val_loader:
