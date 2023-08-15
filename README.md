# Login to Server

CVL Server:

```
ssh aarslan@robustus.ee.ethz.ch
```

AIT:

```
ssh aarslan@ait-server-03
```

Euler:

```
ssh aarslan@euler.ethz.ch
```

# Clone this repository

```
cd

git clone git@github.com:ardarslan/mq.git
```

# Check resource availability

CVL Server:

```
grep --color=always --extended-regexp 'free|$' /home/sladmcvl/smon.txt
```

# List folders, hidden folders, files and hidden files and show the largest one at the bottom

```
du -sch .[!.]* * | sort -h
```

# Start a VS Code Server

CVL Server:

```
srun --time 720 --gres=gpu:1 --cpus-per-task=1 --mem=10G --pty bash -i

OVS_HOST=$(hostname -f) && openvscode-server --host $OVS_HOST --port 5900 --accept-server-license-terms --telemetry-level off |sed "s/localhost/$OVS_HOST/g"
```

Euler:

Go to https://jupyter.euler.hpc.ethz.ch in your browser.

AIT:

Use remote-ssh extension of your local VS Code.

# Start a Jupyter Notebook

```
cd $CODE

jupyter notebook --no-browser --port 5998 --ip $(hostname -f)
```

# Debug Baseline Repository

Import "pdb" package in the files that you want to add a breakpoint to.

Add "pdb.set_trace()" to the lines that you want to add a breakpoint at.

```
srun --time 720 --gres=gpu:1 --cpus-per-task=1 --nodelist=biwirender05 --pty bash -i

mamba deactivate

mamba activate mq_model

cd $CODE/scripts/07_reproduce_baseline_results

python -m pdb train.py

c

```

Use "c" to continue until the first breakpoint.

Use "n" to make one step.

Use "s" to step in.

# Go into mq folder

```
cd $CODE
```

# 01_01 - Update ~/.profile and also export them

```
export LC_ALL=C.UTF-8

export AM_I_DOCKER=False

export BUILD_WITH_CUDA=True
```

CVL Server:

```
export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs02/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
```

Euler:

```
export CODE=/cluster/home/aarslan/mq

export SCRATCH=/cluster/scratch/aarslan
```

AIT:

```
export CODE=/local/home/aarslan/mq

export SCRATCH=/data/aarslan

export CUDA_HOME=/usr/local/cuda
```

# 01_02 - Install package manager

cd $CODE

wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh

chmod +x Mambaforge-Linux-x86_64.sh

rm -rf $SCRATCH/mambaforge

./Mambaforge-Linux-x86_64.sh (Use $SCRATCH/mambaforge for mambaforge path)

rm -rf Mambaforge-Linux-x86_64.sh

exit

# 01_03 - Install MQ data packages

Open a new terminal.
```
mamba deactivate

rm -rf $SCRATCH/mambaforge/envs/mq_data

mamba create -n mq_data python=3.9.9

mamba activate mq_data

cd $CODE/scripts/01_setup_environment/


rm -rf ~/.cache

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

mkdir $SCRATCH/pip_temp

export TMPDIR=$SCRATCH/pip_temp

mkdir $SCRATCH/pip_cache

export PIP_CACHE_DIR=$SCRATCH/pip_cache

pip install --upgrade pip

(
For CVL and Euler:

mamba deactivate

module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy

mamba activate mq_data

cd $CODE/scripts/01_setup_environment

chmod +x install_torch_torchvision.sh

sbatch --gpus=1 --cpus-per-task=2 --mem-per-cpu=16G --gres=gpumem:10g install_torch_torchvision.sh
)

(
For AIT:

mamba deactivate

module load cuda/11.3

mamba activate mq_data

cd $CODE/scripts/01_setup_environment

chmod +x install_torch_torchvision.sh

./install_torch_torchvision.sh
)

pip install mmcv-full==1.6.0

cd $CODE/scripts/04_extract_frame_features/ego_hos/ego_hos/mmsegmentation

pip install -v -e .

cd $CODE/scripts/04_extract_frame_features/gsam

rm -rf gsam

git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git gsam

cd gsam

git submodule update --init --recursive

pip install -e GroundingDINO

rm -rf .git

sed -i -e 's/from models/from gsam.gsam.Tag2Text.models/g' -e 's/from data/from gsam.gsam.Tag2Text.data/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/models/tag2text.py

sed -i 's/from models/from gsam.gsam.Tag2Text.models/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/models/utils.py

sed -i 's/from models/from gsam.gsam.Tag2Text.models/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/inference_ram.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/datasets/transforms.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/groundingdino.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/util/utils.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/backbone/backbone.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/backbone/position_encoding.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/transformer.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' $CODE/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py

touch $CODE/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/models/__init__.py

rm -rf $CODE/scripts/04_extract_frame_features/gsam/Tag2Text/.git

rm -rf $CODE/scripts/04_extract_frame_features/gsam/VISAM

rm -rf $CODE/scripts/04_extract_frame_features/gsam/grounded-sam-osx

mkdir $SCRATCH/mq_libs

cd $SCRATCH/mq_libs

git clone https://github.com/facebookresearch/detectron2.git

pip install -e detectron2

sed -i 's/(point_indices \/\/ W)/torch.div(point_indices, W, rounding_mode="floor")/g' $SCRATCH/mq_libs/detectron2/projects/PointRend/point_rend/point_features.py

cd $CODE/scripts/01_setup_environment

pip install -r mq_data_requirements.txt

pip install $CODE/scripts/04_extract_frame_features/ofa/ofa/transformers

python3 -m ipykernel install --user --name=mq

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

rm -rf ~/.cache

```

# 01_04 - Install MQ model packages

rm -rf ~/.cache

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

mkdir $SCRATCH/pip_temp

export TMPDIR=$SCRATCH/pip_temp

mkdir $SCRATCH/pip_cache

export PIP_CACHE_DIR=$SCRATCH/pip_cache

mamba create -n mq_model python=3.8

(
For CVL and Euler:

mamba deactivate

module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy

mamba activate mq_model

cd $CODE/scripts/01_setup_environment

chmod +x install_torch_torchvision.sh

sbatch --gpus=1 install_torch_torchvision.sh
)

(
For AIT:

mamba deactivate

module load cuda/11.3

mamba activate mq_model

cd $CODE/scripts/01_setup_environment

chmod +x install_torch_torchvision.sh

./install_torch_torchvision.sh
)

pip install --upgrade pip

cd $CODE/scripts/01_setup_environment

pip install -r mq_model_requirements.txt

cd $CODE/scripts/07_reproduce_baseline_results/libs/utils

python3 setup.py install

# 01_05 - Install MQ visualization packages

cd $CODE/scripts/01_setup_environment

pip install -r mq_visualization_requirements.txt

# 02_01 - Download Ego4D dataset and pre-extracted features

Follow the instructions in scripts/02_download_data/01_download_ego4d_dataset_and_clip_features.txt

# 02_02 - Download pre-trained models of frame feature extractors

```
cd $CODE/scripts/02_download_data

chmod +x 02_download_pretrained_models_of_frame_feature_extractors.sh

./02_download_pretrained_models_of_frame_feature_extractors.sh
```

# 03 - Analyze annotations

Implemented in $CODE/scripts/03_analyze_data/check_annotation_distribution.ipynb

# 04 - Extract frame features

CVL:

```
cd $CODE/scripts/04_extract_frame_features

mamba deactivate

mamba activate mq_data

sbatch --time 720 --gres=gpu:4 --cpus-per-task 4 --mem 50G main.sh -f "<FRAME_FEATURE_NAME>" -q "<QUARTER_INDEX>" -c "<CUDA_VISIBLE_DEVICES>"
```

Euler:

```
cd $CODE/scripts/04_extract_frame_features

mamba deactivate

module load eth_proxy

mamba activate mq_data

sbatch --time=1440 --gpus=1 --cpus-per-task=2 --mem-per-cpu=50G --wrap="./main.sh -f \"blip_vqa\" -q \"1\" -c \"0\""
```

AIT:

```
screen

cd $CODE/scripts/04_extract_frame_features

mamba deactivate

module load cuda/11.3

mamba activate mq_data

chmod +x main.sh

./main.sh -f "<FRAME_FEATURE_NAME>" -q "<QUARTER_INDEX>" -c "<CUDA_VISIBLE_DEVICES>"
```

# 05 - Visualize frame features

cd ~/mq/scripts/05_visualize_frame_features

django-admin startproject djangoproject .

python manage.py migrate

python manage.py runserver 5960

Go to http://bmiccomp01.ee.ethz.ch:5960/ in your browser.

<!-- (NOT IMPLEMENTED YET)

cd scripts/05_visualize_frame_features/ag_thesis_230716/webserver

python3 manage.py migrate
python3 manage.py runserver 5999 -->

# 06 - Analyze frame features

(NOT IMPLEMENTED YET)

# 07 - Reproduce baseline results (Works in CVL Server)

mamba deactivate

mamba activate mq_model

cd $CODE/scripts/07_reproduce_baseline_results

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --nodelist=biwirender07 train.sh

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --nodelist=biwirender07 test.sh

python merge_submission.py


From terminal of your computer run the following lines:

```
cd ~/Desktop

scp aarslan@robustus.ee.ethz.ch:/home/aarslan/mq/scripts/07_reproduce_baseline_results/submission_final.json submission_final.json
```

Login to https://eval.ai/auth/login

Submit submission_final.json to https://eval.ai/web/challenges/challenge-page/1626/leaderboard.

# 08 - Reproduce our results

(NOT IMPLEMENTED YET)
