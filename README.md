# Login to Server

CVL Server:

```
ssh aarslan@robustus.ee.ethz.ch
```

AIT Server:

```
ssh aarslan@ait-server-03
```

# Clone this repository

```
cd

git clone git@github.com:ardarslan/mq.git
```

# Update paths

CVL Server:

```
cd ~/mq

find . -type f -exec sed -i 's/\/srv\/beegfs02\/scratch\/aarslan_data\/data/\/data\/aarslan/g' {} +

find . -type f -exec sed -i 's/\/home\/aarslan\/mq/\/local\/home\/aarslan\/mq/g' {} +

```


AIT Server:

```
cd ~/mq

find . -type f -exec sed -i 's/\/data\/aarslan/\/srv\/beegfs02\/scratch\/aarslan_data\/data/g' {} +

find . -type f -exec sed -i 's/\/local\/home\/aarslan\/mq/\/home\/aarslan\/mq/g' {} +
```

# Check resource availability

CVL Server:

```
grep --color=always --extended-regexp 'free|$' /home/sladmcvl/smon.txt
```

AIT Server:

```
nvidia-smi
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

AIT Server:

Use remote-ssh extension of your local VS Code.

# Start a Jupyter Notebook

```
cd ~/mq

jupyter notebook --no-browser --port 5998 --ip $(hostname -f)
```

# Go into mq folder

```
cd ~/mq
```

# 01_01 - Update ~/.profile and also export them

```
export LC_ALL=C.UTF-8

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export AM_I_DOCKER=False

export BUILD_WITH_CUDA=True
```

CVL Server:
```
export TMPDIR=/home/aarslan/mq/tmp

export SCRATCH=/srv/beegfs02/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
```

AIT Server:
```
export TMPDIR=/local/home/aarslan/mq/tmp

export SCRATCH=/data/aarslan

export CUDA_HOME=/usr/local/cuda
```

# 01_02 - Install package manager

cd ~/mq

wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh

chmod +x Mambaforge-Linux-x86_64.sh

rm -rf /srv/beegfs02/scratch/aarslan_data/data/mambaforge

./Mambaforge-Linux-x86_64.sh (Use /srv/beegfs02/scratch/aarslan_data/data/mambaforge for mambaforge path)

rm -rf Mambaforge-Linux-x86_64.sh

exit

(Open a new terminal)

# 01_03 - Install packages

Open a new terminal.

```
mamba deactivate

rm -rf $SCRATCH/mambaforge/envs/mq

mamba create -n mq python=3.9.9

mamba activate mq

cd ~/mq/scripts/01_setup_environment/

chmod +x 03_install_packages.sh

./03_install_packages.sh
```

Windows + Shift + P

Python: Clear Cache and Reload Window

# 02_01 - Download Ego4D dataset and pre-extracted features

Follow the instructions in scripts/02_download_data/01_download_ego4d_dataset_and_clip_features.txt

# 02_02 - Download pre-trained models of frame feature extractors

```
cd ~/mq/scripts/02_download_data

chmod +x 02_download_pretrained_models_of_frame_feature_extractors.sh

./02_download_pretrained_models_of_frame_feature_extractors.sh
```

# 03 - Check annotation distribution

Implemented in ~/mq/scripts/03_analyze_data/check_annotation_distribution.ipynb

# 04 - Extract frame features

CVL Server:

```
cd ~/mq/scripts/04_extract_frame_features

sbatch --time 720 --gres=gpu:4 --cpus-per-task 4 --mem 50G main.sh -f "<FRAME_FEATURE_NAME>" -q "<QUARTER_INDEX>"
```

AIT Server:

```
cd ~/mq/scripts/04_extract_frame_features

screen

chmod +x main.sh

./main.sh -f "<FRAME_FEATURE_NAME>" -q "<QUARTER_INDEX>"
```

# 05 - Visualize frame features

(NOT IMPLEMENTED YET)

cd scripts/05_visualize_frame_features/ag_thesis_230716/webserver

python3 manage.py migrate
python3 manage.py runserver 5999

# 06 - Analyze frame features

(NOT IMPLEMENTED YET)

# 07 - Reproduce baseline results

python3 ~/mq/scripts/07_reproduce_baseline_results/ego4d_asl/convert_annotation.py --input_annotation_folder_path $SCRATCH/ego4d_data/v2/annotations/ --video_features_folder_path $SCRATCH/ego4d_data/v2/ --ego4d_json_path $SCRATCH/ego4d_data/ego4d.json --output_annotation_file_path ~/mq/scripts/07_reproduce_baseline_results/ego4d_asl/data/ego4d/ego4d_clip_annotations_v3.json

cd ~/mq/scripts/07_reproduce_baseline_results/ego4d_asl

python3 train.py configs/baseline.yaml --combine_train

# 08 - Reproduce our results

(NOT IMPLEMENTED YET)
