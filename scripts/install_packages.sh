pip config set install.user false

mkdir -p $SCRATCH/pip_cache/

pip config set global.cache-dir $SCRATCH/pip_cache

pip install --upgrade pip

for line in $(cat requirements.txt); do pip install $line; done

mim install mmcv-full==1.6.0

cd ~/mq/frame_feature_extractors/ego_hos/ego_hos/mmsegmentation

pip install -v -e .

cd ~/mq/frame_feature_extractors/gsam

rm -rf gsam

git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git gsam

cd gsam

git submodule update --init --recursive

pip install -e GroundingDINO

rm -rf .git

sed -i -e 's/from models/from gsam.gsam.Tag2Text.models/g' -e 's/from data/from gsam.gsam.Tag2Text.data/g' /home/aarslan/mq/frame_feature_extractors/gsam/gsam/Tag2Text/models/tag2text.py

sed -i 's/from models/from gsam.gsam.Tag2Text.models/g' ~/mq/frame_feature_extractors/gsam/gsam/Tag2Text/models/utils.py

sed -i 's/from models/from gsam.gsam.Tag2Text.models/g' ~/mq/frame_feature_extractors/gsam/gsam/Tag2Text/inference_ram.py

touch ~/mq/frame_feature_extractors/gsam/gsam/Tag2Text/models/__init__.py

rm -rf ~/mq/frame_feature_extractors/gsam/Tag2Text/.git

rm -rf ~/mq/frame_feature_extractors/gsam/VISAM

rm -rf ~/mq/frame_feature_extractors/gsam/grounded-sam-osx

pip install ~/mq/frame_feature_extractors/ofa/ofa/transformers

python3 -m ipykernel install --user --name=mq

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

rm -rf ~/.cache