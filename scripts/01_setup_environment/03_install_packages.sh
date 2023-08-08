mamba deactivate

rm -rf $SCRATCH/mambaforge/envs/mq

mamba create -n mq python=3.9.9

mamba activate mq

rm -rf ~/.cache

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

mkdir $SCRATCH/pip_temp

export TMPDIR=$SCRATCH/pip_temp

mkdir $SCRATCH/pip_cache

export PIP_CACHE_DIR=$SCRATCH/pip_cache

pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu115

pip install mmcv-full==1.6.0

cd ~/mq/scripts/04_extract_frame_features/ego_hos/ego_hos/mmsegmentation

pip install -v -e .

cd ~/mq/scripts/04_extract_frame_features/gsam

rm -rf gsam

git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git gsam

cd gsam

git submodule update --init --recursive

pip install -e GroundingDINO

rm -rf .git

sed -i -e 's/from models/from gsam.gsam.Tag2Text.models/g' -e 's/from data/from gsam.gsam.Tag2Text.data/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/models/tag2text.py

sed -i 's/from models/from gsam.gsam.Tag2Text.models/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/models/utils.py

sed -i 's/from models/from gsam.gsam.Tag2Text.models/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/inference_ram.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/datasets/transforms.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/groundingdino.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/util/utils.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/backbone/backbone.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/backbone/position_encoding.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/transformer.py

sed -i 's/from groundingdino/from gsam.gsam.GroundingDINO.groundingdino/g' ~/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py

touch ~/mq/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/models/__init__.py

rm -rf ~/mq/scripts/04_extract_frame_features/gsam/Tag2Text/.git

rm -rf ~/mq/scripts/04_extract_frame_features/gsam/VISAM

rm -rf ~/mq/scripts/04_extract_frame_features/gsam/grounded-sam-osx

cd /srv/beegfs02/scratch/aarslan_data/data/mq_libs

git clone https://github.com/facebookresearch/detectron2.git

pip install -e detectron2

cd ~/mq/scripts/01_setup_environment

pip install -r requirements.txt

pip install ~/mq/scripts/04_extract_frame_features/ofa/ofa/transformers

cd ~/mq/scripts/07_reproduce_baseline_results/ego4d_asl/libs/utils

python3 setup.py install --user

python3 -m ipykernel install --user --name=mq

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

rm -rf ~/.cache

export TMPDIR=/home/aarslan/mq/tmp
