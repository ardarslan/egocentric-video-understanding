rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

rm -rf ~/.cache

pip config set install.user false

mkdir -p $SCRATCH/pip_cache/

pip config set global.cache-dir $SCRATCH/pip_cache

pip install --upgrade pip

for line in $(cat requirements.txt); do pip install $line; done

mim install mmcv-full==1.6.0

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

touch ~/mq/scripts/04_extract_frame_features/gsam/gsam/Tag2Text/models/__init__.py

rm -rf ~/mq/scripts/04_extract_frame_features/gsam/Tag2Text/.git

rm -rf ~/mq/scripts/04_extract_frame_features/gsam/VISAM

rm -rf ~/mq/scripts/04_extract_frame_features/gsam/grounded-sam-osx

pip install ~/mq/scripts/04_extract_frame_features/ofa/ofa/transformers

python3 ~/mq/scripts/07_reproduce_baseline_results/ego4d_asl/convert_annotation.py --input_annotation_folder_path $SCRATCH/ego4d_data/v2/annotations/ --video_features_folder_path $SCRATCH/ego4d_data/v2/ --ego4d_json_path $SCRATCH/ego4d_data/ego4d.json --output_annotation_file_path ~/mq/scripts/07_reproduce_baseline_results/ego4d_asl/data/ego4d/ego4d_clip_annotations_v3.json

cd ~/mq/scripts/07_reproduce_baseline_results/ego4d_asl/libs/utils

python3 setup.py install --user

python3 -m ipykernel install --user --name=mq

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

rm -rf ~/.cache
