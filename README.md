# Login to CVL Server

```
ssh aarslan@robustus.ee.ethz.ch
```

# Check resource availability

```
grep --color=always --extended-regexp 'free|$' /home/sladmcvl/smon.txt
```

# List folders, hidden folders, files and hidden files and show the largest one at the bottom

```
du -sch .[!.]* * | sort -h
```

# Start a VS Code Server

```
srun --time 720 --gres=gpu:5 --cpus-per-task=5 --mem=40G --pty bash -i

OVS_HOST=$(hostname -f) && openvscode-server --host $OVS_HOST --port 5900 --accept-server-license-terms --telemetry-level off |sed "s/localhost/$OVS_HOST/g"
```

# Start a Jupyter Notebook

```
cd ~/mq

jupyter notebook --no-browser --port 5998 --ip $(hostname -f)
```

# Go into mq folder

```
cd mq
```

# 01_01 - Update ~/.profile and ~/.bashrc files

```
chmod +x scripts/01_setup_environment/01_update_profile_and_bashrc.sh

./scripts/01_setup_environment/01_update_profile_and_bashrc.sh
```

# 01_02 - Install pyenv

```
chmod +x scripts/01_setup_environment/02_install_pyenv.sh

./scripts/01_setup_environment/02_install_pyenv.sh

exit
```

# 01_03 - Install packages

Open a new terminal.

```
chmod +x scripts/01_setup_environment/03_install_packages.sh

./scripts/01_setup_environment/03_install_packages.sh
```

Windows + Shift + P

Python: Clear Cache and Reload Window

# 02_01 - Download Ego4D dataset and pre-extracted features

Follow the instructions in scripts/02_download_data/01_download_ego4d_dataset_and_clip_features.txt

# 02_02 - Download pre-trained models of frame feature extractors

```
chmod +x scripts/02_download_data/02_download_pretrained_models_of_frame_feature_extractors.sh

./scripts/02_download_data/02_download_pretrained_models_of_frame_feature_extractors.sh
```

# 03 - Check annotation distribution

(NOT IMPLEMENTED YET)

# 04 - Extract frame features

```

cd ~/mq/scripts/03_extract_frame_features/ofa

pip install ofa/transformers

screen

cd ~/mq/scripts/04_extract_frame_features

python3 main.py --frame_feature_name unidet

python3 main.py --frame_feature_name visor_hos

python3 main.py --frame_feature_name ego_hos

python3 main.py --frame_feature_name gsam

python3 main.py --frame_feature_name ofa

python3 main.py --frame_feature_name blip_captioning

python3 main.py --frame_feature_name blip_vqa
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
