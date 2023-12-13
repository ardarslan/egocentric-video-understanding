#!/bin/bash

mkdir $SCRATCH/mq_libs

cd $SCRATCH/mq_libs

# gsam
mkdir gsam

wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O gsam/groundingdino_swint_ogc.pth

wget https://huggingface.co/spaces/xinyu1205/Tag2Text/resolve/main/ram_swin_large_14m.pth -O gsam/ram_swin_large_14m.pth

# ego_hos
mkdir -p ego_hos/seg_twohands_ccda

gdown 1TF5VCkWKZG6IRgVpnpiwA2P4GHx-0Drc -O ego_hos/seg_twohands_ccda/best_mIoU_iter_56000.pth

mkdir -p ego_hos/twohands_to_cb_ccda

gdown 1dKnwQYF-3TeyoyfmbFAw8tt1TfhUq4Qj -O ego_hos/twohands_to_cb_ccda/best_mIoU_iter_76000.pth

mkdir -p ego_hos/twohands_cb_to_obj2_ccda

gdown 1JdewAV1XJyR9reVxEVrwwCH-MwQD6Aqy -O ego_hos/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth

# blip2-flan-t5-xl
git clone https://huggingface.co/Salesforce/blip2-flan-t5-xl

cd blip2-flan-t5-xl

wget https://huggingface.co/Salesforce/blip2-flan-t5-xl/resolve/main/pytorch_model-00001-of-00002.bin

wget https://huggingface.co/Salesforce/blip2-flan-t5-xl/resolve/main/pytorch_model-00002-of-00002.bin

# word2vec
mkdir word2vec

cd word2vec

gdown 0B7XkCwpI5KDYNlNUTTlSS21pQmM -O GoogleNews-vectors-negative300.bin.gz

gunzip -k GoogleNews-vectors-negative300.bin.gz

rm -rf GoogleNews-vectors-negative300.bin.gz

cd ..

# java

mkdir java

wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz -O java/jdk-21_linux-x64_bin.tar.gz

cd java

tar -xvf jdk-21_linux-x64_bin.tar.gz

export JAVA_HOME=jdk-install-dir

export PATH=$JAVA_HOME/bin:$PATH

# stanford-corenlp

wget https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.5.5.zip -O stanford-corenlp-4.5.5.zip

unzip stanford-corenlp-4.5.5.zip

mv stanford-corenlp-4.5.5 stanford-corenlp

rm -rf stanford-corenlp-4.5.5.zip

wget https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.5.5-models-english.jar -O stanford-corenlp/stanford-corenlp-4.5.5-models-english.jar

# video-blip-flan-t5-xl-ego4d

cd $SCRATCH/mq_libs

git clone https://huggingface.co/kpyu/video-blip-flan-t5-xl-ego4d

cd video-blip-flan-t5-xl-ego4d

rm -rf pytorch_model-00001-of-00002.bin

rm -rf pytorch_model-00002-of-00002.bin

wget https://huggingface.co/kpyu/video-blip-flan-t5-xl-ego4d/resolve/main/pytorch_model-00001-of-00002.bin

wget https://huggingface.co/kpyu/video-blip-flan-t5-xl-ego4d/resolve/main/pytorch_model-00002-of-00002.bin
