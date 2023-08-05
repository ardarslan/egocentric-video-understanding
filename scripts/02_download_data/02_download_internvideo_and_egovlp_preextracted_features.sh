wget https://zenodo.org/record/7340838/files/ego4d_mq_videointern_eccv2022_verb_features.zip -O $SCRATCH/ego4d_data/v2/internvideo.zip

python3 /home/aarslan/mq/scripts/02_download_data/unzipper.py --zip_file_path $SCRATCH/ego4d_data/v2/internvideo.zip --unzipped_folder_path $SCRATCH/ego4d_data/v2/internvideo

rm -rf $SCRATCH/ego4d_data/v2/internvideo.zip

gdown 1CDRV0FIMXp3wB5Q1o_UcXla7Xw7P5ORG -O $SCRATCH/ego4d_data/v2/egovlp_egonce.zip

python3 /home/aarslan/mq/scripts/02_download_data/unzipper.py --zip_file_path $SCRATCH/ego4d_data/v2/egovlp_egonce.zip --unzipped_folder_path $SCRATCH/ego4d_data/v2/egovlp_egonce

rm -rf $SCRATCH/ego4d_data/v2/egovlp_egonce.zip