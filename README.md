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

git clone git@github.com:ardarslan/moments-querying.git

mv moments-querying mq
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
export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

cd $CODE

srun --time 720 --cpus-per-task=2 --gres=gpu:1 --mem=50G --constraint='geforce_gtx_1080_ti' --pty bash -i

mamba activate mq_model

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

cd $CODE/scripts/08_reproduce_mq_experiments

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

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

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

curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

chmod +x Miniforge3-Linux-x86_64.sh

rm -rf $SCRATCH/mambaforge

./Miniforge3-Linux-x86_64.sh (Use $SCRATCH/miniforge for mambaforge path)

rm -rf Miniforge3-Linux-x86_64.sh

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

mamba install jupyter

python -m ipykernel install --user --name=mq_model

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

cd $CODE/scripts/08_reproduce_mq_experiments/libs/utils

python3 setup.py install

# 01_05 - Install MQ visualization packages

mamba deactivate

mamba create -n mq_visualization python=3.9.9

mamba activate mq_visualization

cd $CODE/scripts/01_setup_environment

python3 -m pip install -r mq_visualization_requirements.txt

# 01_06 - Install MQ BLIP2 caption analysis packages

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

<!-- export CODE=/local/home/aarslan/mq

export SCRATCH=/data/aarslan

export CUDA_HOME=/usr/local/cuda -->

mamba deactivate

mamba create -n mq_blip2_caption_analysis python=3.9.9

mamba activate mq_blip2_caption_analysis

cd $CODE/scripts/01_setup_environment

python3 -m pip install -r mq_blip2_caption_analysis_requirements.txt

<!--
# 01_07 - Install MQ finetune packages

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs02/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

mamba deactivate

export TMPDIR=$SCRATCH/pip_temp

mamba create --name mq_finetune

mamba activate mq_finetune

mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch-nightly -c nvidia

pip install numpy tqdm pillow transformers accelerate opencv-python

cd $CODE/scripts/06_blip2_caption_analysis/06_fine_tune_blip2_frame_wise

python3 01_extract_labels_file.py --split train

python3 01_extract_labels_file.py --split val

sbatch --time 720 --gres=gpu:8 --cpus-per-task 4 --mem-per-cpu 200G 02_fine_tune_blip2_frame_wise.sh -->

# 01_07 - Install MQ BLIP2 embedding analysis packages

AIT
```
export CODE=/local/home/aarslan/mq

export SCRATCH=/data/aarslan

export CUDA_HOME=/usr/local/cuda
```

CVL
```
export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

export TMPDIR=$SCRATCH/pip_temp

rm -rf ~/.cache

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

rm -rf $SCRATCH/mambaforge/envs/mq_blip2_embedding_analysis

mkdir $SCRATCH/pip_temp

mkdir $SCRATCH/pip_cache

mamba create -n mq_blip2_embedding_analysis python=3.11

mamba activate mq_blip2_embedding_analysis

cd $CODE

wget https://bootstrap.pypa.io/get-pip.py

python3 get-pip.py --user

rm -rf get-pip.py

python3 -m pip install pipx

python3 -m pipx ensurepath

exit

Open a new terminal

mamba activate mq_blip2_embedding_analysis

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

export TMPDIR=$SCRATCH/pip_temp

cd $CODE/scripts/07_blip2_embedding_analysis/

cd EILEV

pip install -e .

cd $CODE/scripts/07_blip2_embedding_analysis/transformers

pip install -e .
```

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

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

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

# 06 - BLIP2 Caption Analysis

```
<!-- export CODE=/local/home/aarslan/mq

export SCRATCH=/data/aarslan

export CUDA_HOME=/usr/local/cuda -->

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

mamba activate mq_blip2_caption_analysis

cd $CODE/scripts/06_blip2_caption_analysis/01_extract_blip2_answer_dependency_parsing_features

chmod +x 01_extract_blip2_answer_dependency_parsing_features.sh

sbatch --time 720 --cpus-per-task=24 --mem 40G 01_extract_blip2_answer_dependency_parsing_features.sh

cd $CODE/scripts/06_blip2_caption_analysis/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features

./01_process_ground_truth_labels.sh

./02_process_asl_predictions.sh

./03_blip2_dictionary_matching.sh

./03_blip2_sbert_matching.sh -q 0 -c 4 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 1 -c 5 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 2 -c 6 -b sentence-transformers/all-distilroberta-v1

./03_blip2_sbert_matching.sh -q 3 -c 7 -b sentence-transformers/all-distilroberta-v1


./04_max_per_question_per_label_postprocessing.sh -p asl_ego4d_features

./04_max_per_question_per_label_postprocessing.sh -p asl_predictions

./04_max_per_question_per_label_postprocessing.sh -p blip2_dictionary_matching_predictions

./04_max_per_question_per_label_postprocessing.sh -p blip2_sbert_matching_all-distilroberta-v1_predictions


(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h 0.2 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h 0.4 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h 0.6 -s val



(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h 0.8 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h 1.0 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h max -s val


(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v2_max_per_label_predictions -t no_temporal_aggregation -h 0.2 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v2_max_per_label_predictions -t no_temporal_aggregation -h 0.4 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v2_max_per_label_predictions -t no_temporal_aggregation -h 0.6 -s val


(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v2_max_per_label_predictions -t no_temporal_aggregation -h 0.8 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v2_max_per_label_predictions -t no_temporal_aggregation -h 1.0 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p proposed_features_v2_max_per_label_predictions -t no_temporal_aggregation -h max -s val


(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p asl_ego4d_features_max_per_label_predictions -t no_temporal_aggregation -h 0.2 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p asl_ego4d_features_max_per_label_predictions -t no_temporal_aggregation -h 0.4 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p asl_ego4d_features_max_per_label_predictions -t no_temporal_aggregation -h 0.6 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p asl_ego4d_features_max_per_label_predictions -t no_temporal_aggregation -h 0.8 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p asl_ego4d_features_max_per_label_predictions -t no_temporal_aggregation -h 1.0 -s val

(RUNNING)
sbatch --time 720 --gres=gpu:0 --cpus-per-task=4 --mem 50G 05_evaluate_predictions.sh -p asl_ego4d_features_max_per_label_predictions -t no_temporal_aggregation -h max -s val




<!--
(RUNNING)
./05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h 0.2 -s val

(RUNNING)
./05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h 0.4 -s val

(RUNNING)
./05_evaluate_predictions.sh -p proposed_features_v5_max_per_label_predictions -t no_temporal_aggregation -h 0.6 -s val -->



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
```

# 07 - BLIP2 Embedding Analysis

AIT

```
screen

cd $CODE/scripts/07_blip2_embedding_analysis

mamba activate mq_blip2_embedding_analysis

(DONE)
CUDA_VISIBLE_DEVICES=2,6 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 0 --split train

(DONE)
CUDA_VISIBLE_DEVICES=1,5 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 1 --split train

(RUNNING)
CUDA_VISIBLE_DEVICES=2,6 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 2 --split train

(RUNNING)
CUDA_VISIBLE_DEVICES=1,5 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 3 --split train


(NOT DONE)
CUDA_VISIBLE_DEVICES=2,6 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 4 --split train

(NOT DONE)
CUDA_VISIBLE_DEVICES=1,5 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 5 --split train

(NOT DONE)
CUDA_VISIBLE_DEVICES=2,6 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 6 --split train

(NOT DONE)
CUDA_VISIBLE_DEVICES=1,5 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 7 --split train

(NOT DONE)
CUDA_VISIBLE_DEVICES=2,6 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 8 --split train

(NOT DONE)
CUDA_VISIBLE_DEVICES=1,5 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 9 --split train

(NOT DONE)
CUDA_VISIBLE_DEVICES=2,6 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 10 --split train

(NOT DONE)
CUDA_VISIBLE_DEVICES=1,5 python3 01_extract_frame_features.py --frame_feature_name blip2_vqa --quarter_index 11 --split train

```

CVL

```
export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

mamba activate mq_blip2_embedding_analysis

cd $CODE/scripts/07_blip2_embedding_analysis


(DONE)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 0 -s val

(DONE)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 1 -s val

(DONE)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 2 -s val

(DONE)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 3 -s val

(RUNNING)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 4 -s val

(RUNNING)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 5 -s val

(RUNNING)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 6 -s val

(RUNNING)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 7 -s val

(NOT DONE)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 8 -s val

(NOT DONE)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 9 -s val

(NOT DONE)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 10 -s val

(NOT DONE)
sbatch --time 720 --cpus-per-task 2 --gres=gpu:2 --mem-per-cpu 50G 01_extract_frame_features.sh -f blip2_vqa -q 11 -s val

```

# 08_01 - Reproduce baseline results (Works in CVL Server, Without Ensemble)

```

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

mamba deactivate

mamba activate mq_model

cd $CODE/scripts/08_reproduce_mq_experiments

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 60G train.sh

sbatch --time 720 --gres=gpu:1 --cpus-per-task=5 --mem 50G test.sh

python merge_submission.py
```

From terminal of your computer run the following lines:

```
cd ~/Desktop

scp aarslan@robustus.ee.ethz.ch:/home/aarslan/mq/scripts/08_reproduce_mq_experiments/proposed_features_v2.json proposed_features_v2.json
```

Login to https://eval.ai/auth/login

Submit asl_original_predictions.json to https://eval.ai/web/challenges/challenge-page/1626/leaderboard

# 08_02 - Reproduce our results

(NOT IMPLEMENTED YET)

# 08_03 - Evaluate on validation split

export CODE=/home/aarslan/mq

export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf

export SCRATCH=/srv/beegfs-benderdata/scratch/aarslan_data/data

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

mamba activate mq_model

cd $CODE/scripts/08_reproduce_mq_experiments

chmod +x val.sh

sbatch --time 720 --gres=gpu:2 --cpus-per-task 5 --mem-per-cpu 200G val.sh


# Postprocess Data

mamba activate mq_model



sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 0
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 1
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 2
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 3
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 4
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 5
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 6
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 7
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 8
sbatch --time 720 --cpus-per-task 1 --gres=gpu:0 --mem-per-cpu 50G postprocess_data.sh -q 9

sbatch --time 720 --cpus-per-task 4 --gres=gpu:0 --mem-per-cpu 50G incremental_pca.sh
