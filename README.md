# Login to Server

CVL Server:

```
ssh aarslan@robustus.ee.ethz.ch
```

AIT:

```
ssh aarslan@ait-server-03
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
srun --time 720 --cpus-per-task=1 --mem=10G --pty bash -i

OVS_HOST=$(hostname -f) && openvscode-server --host $OVS_HOST --port 5900 --accept-server-license-terms --telemetry-level off |sed "s/localhost/$OVS_HOST/g"
```

AIT:

Use remote-ssh extension of your local VS Code.

# Start a Jupyter Notebook

CVL

```
cd $CODE

srun --time 720 --cpus-per-task=2 --gres=gpu:1 --mem=50G --constraint='geforce_gtx_1080_ti' --pty bash -i

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs02/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

mamba activate mq_analysis

jupyter notebook --no-browser --port 5998 --ip $(hostname -f)
```

AIT

```
http://127.0.0.1:5961/tree?token=bd667dd4f417482951fa1ec99fcc9521adc4e6ea26f4a9c7
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

export JAVA_HOME=$SCRATCH/mq_libs/java

export PATH=$JAVA_HOME/bin:$PATH
```


AIT:

```
export CODE=/local/home/aarslan/mq

export SCRATCH=/data/aarslan

export CUDA_HOME=/usr/local/cuda

export JAVA_HOME=$SCRATCH/mq_libs/java

export PATH=$JAVA_HOME/bin:$PATH
```

# 01_02 - Install package manager

cd $CODE

wget https://github.com/mamba-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh

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
For CVL:

mamba deactivate

mamba activate mq_data

cd $CODE/scripts/01_setup_environment

chmod +x install_torch_torchvision.sh

sbatch --gres=gpu:1 install_torch_torchvision.sh
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

cd $CODE/scripts/01_setup_environment

pip install -r mq_data_requirements.txt

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
For CVL:

mamba deactivate

mamba activate mq_model

cd $CODE/scripts/01_setup_environment

python3 -m pip install torch==1.12 torchvision==0.13 --index-url https://download.pytorch.org/whl/cu113
)

(
For AIT:

mamba deactivate

module load cuda/11.3

mamba activate mq_model

cd $CODE/scripts/01_setup_environment

python3 -m pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --index-url https://download.pytorch.org/whl/cu113
)

python3 -m pip install --upgrade pip

cd $CODE/scripts/01_setup_environment

python3 -m pip install -r mq_model_requirements.txt

cd $CODE/scripts/07_reproduce_mq_experiments/libs/utils

python3 setup.py install

# 01_05 - Install MQ visualization packages

mamba deactivate

mamba create -n mq_visualization python=3.9.9

mamba activate mq_visualization

cd $CODE/scripts/01_setup_environment

python3 -m pip install -r mq_visualization_requirements.txt

# 01_06 - Install MQ analysis packages

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs02/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

mamba deactivate

mamba create -n mq_analysis python=3.9.9

mamba activate mq_analysis

cd $CODE/scripts/01_setup_environment

python3 -m pip install -r mq_analysis_requirements.txt

# 01_07 - Install MQ finetune packages

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs02/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

<!-- export CODE=/local/home/aarslan/mq

export SCRATCH=/data/aarslan

export CUDA_HOME=/usr/local/cuda -->

mamba deactivate

export TMPDIR=$SCRATCH/pip_temp

mamba create --name mq_finetune

mamba activate mq_finetune

mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia

pip install numpy tqdm pillow transformers accelerate opencv-python

cd $CODE/scripts/06_analyze_frame_features/06_fine_tune_blip2_frame_wise

python3 01_extract_labels_file.py --split train

python3 01_extract_labels_file.py --split val

sbatch --time 720 --gres=gpu:6 --cpus-per-task 4 --mem-per-cpu 200G 02_fine_tune_blip2_frame_wise.sh





<!--
CONDA_OVERRIDE_CUDA=11.8 mamba create --name mq_finetune python=3.9.9 gcc=9.4.0 gxx=9.4.0 pytorch torchvision pytorch-cuda --channel pytorch --channel nvidia

mamba activate mq_finetune

mamba install cmake libzlib protobuf zlib lmdb libjpeg-turbo zstd zstd-static openjpeg libtiff opencv av ffmpeg libflac libogg libvorbis libopus libsndfile libtar cfitsio

export TMPDIR=$SCRATCH/pip_temp

mamba install cudatoolkit-dev=11.0

cd $SCRATCH/mq_libs

git clone --recursive https://github.com/NVIDIA/DALI

cd DALI

mkdir build

cd build

vi cmake_command.sh

Copy the following:
```
#!/bin/bash

cmake -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_PREFIX_PATH=$(which protoc) -DCMAKE_CXX_COMPILER=$(which c++) -DCMAKE_C_COMPILER=$(which gcc) -DBUILD_NVJPEG=OFF -DBUILD_LIBSND=OFF -DBUILD_LIBTAR=OFF -DBUILD_JPEG_TURBO=OFF -DCMAKE_BUILD_TYPE=Release ..
```

chmod +x cmake_command.sh

sbatch --time 720 --gres=gpu:2 --cpus-per-task 2 --mem-per-cpu 10G ./cmake_command.sh

make -->

<!--
mamba install lmdb

pip3 install lmdb --upgrade
-->
<!--
exit

ssh aarslan@robustus.ee.ethz.ch

mamba activate mq_analysis

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs02/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit -->

<!-- cd $SCRATCH/mq_libs/DALI_deps/third_part/lmdb/libraries/liblmdb

Modify the line prefix = /usr/local to prefix = /srv/beegfs02/scratch/aarslan_data/data/mq_libs/DALI_deps/third_party/lmdb/libraries/liblmdb

make

make install -->
<!--
cd $SCRATCH/mq_libs/DALI/build

cmake -DCMAKE_CXX_COMPILER="/usr/bin/clang++-7" -DCMAKE_GCC_COMPILER="/usr/bin/gcc" -BUILD_DALI_NODEPS=ON -DCMAKE_BUILD_TYPE=Release ..

python3 -m ipykernel install --user --name=mq_analysis -->

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
export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs02/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

cd $CODE/scripts/04_extract_frame_features

mamba deactivate

mamba activate mq_data

sbatch --time 720 --gres=gpu:2 --cpus-per-task 2 --mem-per-cpu 200G 01_extract_frame_features.sh -f "blip2_vqa" -q "0" -c "0,1"

sbatch --time 720 --gres=gpu:2 --cpus-per-task 2 --mem-per-cpu 200G 01_extract_frame_features.sh -f "blip2_vqa" -q "1" -c "0,1"

sbatch --time 720 --gres=gpu:2 --cpus-per-task 2 --mem-per-cpu 200G 01_extract_frame_features.sh -f "blip2_vqa" -q "2" -c "0,1"

sbatch --time 720 --gres=gpu:2 --cpus-per-task 2 --mem-per-cpu 200G 01_extract_frame_features.sh -f "blip2_vqa" -q "3" -c "0,1"

```

AIT:

```
screen

export CODE=/local/home/aarslan/mq

export SCRATCH=/data/aarslan

export CUDA_HOME=/usr/local/cuda

cd $CODE/scripts/04_extract_frame_features

mamba deactivate

module load cuda/11.3

mamba activate mq_data

chmod +x 01_extract_frame_features.sh

DONE
./01_extract_frame_features.sh -f "blip2_vqa" -q "0" -c "4"

./01_extract_frame_features.sh -f "blip2_vqa" -q "1" -c "5"

./01_extract_frame_features.sh -f "blip2_vqa" -q "2" -c "6"

./01_extract_frame_features.sh -f "blip2_vqa" -q "3" -c "7"


# 05_01 - Extract Frames

```
cd $CODE/scripts/05_visualize_frame_features/01_extract_frames

mamba activate mq_visualization

python3 frame_extractor.py
```

# 05_02 - Stream Video

```
cd $CODE/scripts/05_visualize_frame_features/02_stream_video

mamba activate mq_visualization

python3 manage.py runserver 5960
```

# 05_03 - Horizontal Bar Plots

```
cd $CODE/scripts/05_visualize_frame_features/03_horizontal_bar_plots

mamba activate mq_visualization

python3 horizontal_bar_plots.py --clip_id 02246bfe-dcef-465d-9aa5-47a2e71460dd --port 8050

python3 horizontal_bar_plots.py --clip_id 0076e425-bdb6-48b3-b4d3-695089ac9800 --port 8051

python3 horizontal_bar_plots.py --clip_id 013559ff-eab2-4c25-a475-90bf56f5ae9e --port 8053

python3 horizontal_bar_plots.py --clip_id 00e4af86-adca-479f-a20a-402f1bc933a0 --port 8054

python3 horizontal_bar_plots.py --clip_id 003c5ae8-3abd-4824-8efb-21a9a4f8eafe --port 8055
```

# 06 - Analyze frame features (CVL)

mamba activate mq_analysis

cd $CODE/scripts/06_analyze_frame_features/01_extract_blip2_answer_dependency_parsing_features

chmod +x 01_extract_blip2_answer_dependency_parsing_features.sh

sbatch --time 720 --cpus-per-task=24 --mem 40G 01_extract_blip2_answer_dependency_parsing_features.sh

cd $CODE/scripts/06_analyze_frame_features/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features

sbatch --time 720 --cpus-per-task=24 --mem 200G 01_process_ground_truth_labels.sh

sbatch --time 720 --cpus-per-task=8 --mem 200G 02_process_asl_predictions.sh

sbatch --time 720 --cpus-per-task=8 --mem 40G 03_blip2_dictionary_matching.sh

./03_blip2_sbert_matching.sh -q 0 -c 4 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 1 -c 5 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 2 -c 6 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 3 -c 7 -b sentence-transformers/all-distilroberta-v1

sbatch --time 720 --cpus-per-task=8 --mem 200G ./04_max_per_question_per_label_postprocessing.sh -p blip2_dictionary_matching_predictions

sbatch --time 720 --cpus-per-task=24 --mem 200G ./04_max_per_question_per_label_postprocessing.sh -p blip2_sbert_matching_all-distilroberta-v1_predictions

./05_evaluate_predictions.sh -p blip2_dictionary_matching_max_per_label_predictions -t no_temporal_aggregation -h 0.2 -s val

./05_evaluate_predictions.sh -p blip2_dictionary_matching_max_per_label_predictions -t no_temporal_aggregation -h 0.4 -s val

./05_evaluate_predictions.sh -p blip2_dictionary_matching_max_per_label_predictions -t no_temporal_aggregation -h 0.6 -s val

./05_evaluate_predictions.sh -p blip2_dictionary_matching_max_per_label_predictions -t no_temporal_aggregation -h 0.8 -s val

./05_evaluate_predictions.sh -p blip2_dictionary_matching_max_per_label_predictions -t no_temporal_aggregation -h 1.0 -s val

./05_evaluate_predictions.sh -p blip2_dictionary_matching_max_per_label_predictions -t no_temporal_aggregation -h max -s val



./05_evaluate_predictions.sh -p blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions -t no_temporal_aggregation -h 0.2 -s val

./05_evaluate_predictions.sh -p blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions -t no_temporal_aggregation -h 0.4 -s val

./05_evaluate_predictions.sh -p blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions -t no_temporal_aggregation -h 0.6 -s val

./05_evaluate_predictions.sh -p blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions -t no_temporal_aggregation -h 0.8 -s val

./05_evaluate_predictions.sh -p blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions -t no_temporal_aggregation -h 1.0 -s val

./05_evaluate_predictions.sh -p blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions -t no_temporal_aggregation -h max -s val


# 06 - Analyze frame features (AIT)

screen

mamba activate mq_analysis

cd $CODE/scripts/06_analyze_frame_features/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features

./03_blip2_sbert_matching.sh -q 0 -c 4 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 1 -c 5 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 2 -c 6 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 3 -c 7 -b sentence-transformers/all-distilroberta-v1

./04_max_per_question_per_label_postprocessing.sh -p blip2_sbert_matching_all-distilroberta-v1_predictions



sbatch --time 720 --gres=gpu:4 --cpus-per-task 4 --mem-per-cpu 200G 02_fine_tune_blip2_frame_wise.sh

# 07_01 - Reproduce baseline results (Works in CVL Server, Without Ensemble)

mamba deactivate

mamba activate mq_model

cd $CODE/scripts/07_reproduce_mq_experiments

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 60G train.sh

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 60G test.sh

# 07_02 - Reproduce baseline results (Works in CVL Server, With Ensemble)

mamba deactivate

mamba activate mq_model

cd $CODE/scripts/07_reproduce_baseline_results

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 50G --nodelist=biwirender08 train_1.sh

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 50G --nodelist=biwirender08 train_2.sh

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 50G --nodelist=biwirender08 train_3.sh

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 50G --nodelist=biwirender08 train_4.sh

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 50G --nodelist=biwirender08 train_5.sh

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 50G --nodelist=biwirender08 test.sh

python merge_submission.py


From terminal of your computer run the following lines:

```
cd ~/Desktop

scp aarslan@robustus.ee.ethz.ch:/home/aarslan/mq/scripts/07_reproduce_baseline_results/submission_final.json submission_final.json
```

Login to https://eval.ai/auth/login

Submit submission_final.json to https://eval.ai/web/challenges/challenge-page/1626/leaderboard

# 08 - Reproduce our results

(NOT IMPLEMENTED YET)
