rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

rm -rf ~/.cache

pip config set install.user false

mkdir -p $SCRATCH/pip_temp

export TMPDIR=$SCRATCH/pip_temp

mkdir -p $SCRATCH/pip_cache

pip config set global.cache-dir $SCRATCH/pip_cache

pip install --upgrade pip



pip install torch

pip install torchvision

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

cd ~/mq/scripts/01_setup_environment

for line in $(cat requirements.txt); do pip install $line; done

mim install mmcv-full==1.6.0

cd ~/mq/scripts/04_extract_frame_features/ego_hos/ego_hos/mmsegmentation

pip install -v -e .

pip install ~/mq/scripts/04_extract_frame_features/ofa/ofa/transformers

cd ~/mq/scripts/07_reproduce_baseline_results/ego4d_asl/libs/utils

python3 setup.py install --user

python3 -m ipykernel install --user --name=mq

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

rm -rf ~/.cache

export TMPDIR=/home/aarslan/mq/tmp