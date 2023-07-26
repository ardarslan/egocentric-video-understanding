# Add the following lines to ~/.profile

```
export BEEGFS_SCRATCH=/srv/beegfs02/scratch/aarslan_data/data
export SCRATCH=/scratch/aarslan
export TMPDIR=/scratch/aarslan/pip_temp
export LC_ALL=C.UTF-8
export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf
export PYENV_ROOT="$SCRATCH/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
export PIP_CACHE_DIR=$SCRATCH/pip_cache/
```

# Add the following lines to ~/.bashrc

```
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
function localpip {
    PYTHONUSERBASE=$SCRATCH
    PATH=$PYTHONUSERBASE/bin:$PATH
    export PYTHONUSERBASE PATH
}
```

# Login to CVL Server

```
ssh aarslan@robustus.ee.ethz.ch
```

# Start a VS Code Server

```
srun --time 720 --gres=gpu:4 --cpus-per-task=1 --nodes=4 --mem=10G --pty bash -i

OVS_HOST=$(hostname -f) && openvscode-server --host $OVS_HOST --port 5900-5999 --accept-server-license-terms --telemetry-level off |sed "s/localhost/$OVS_HOST/g"
```

# Install new pyenv

```
rm -rf $SCRATCH/.pyenv

rm -rf $SCRATCH/pip_cache

cd

cd mq

chmod +x pyenv-installer

bash ./pyenv-installer

env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.9

pyenv rehash

pyenv global 3.9.9

pyenv update

exit

```

Open a new terminal.


```
pip config set install.user false
```

# Move pip's cache folder into scratch

```
mkdir -p $SCRATCH/pip_cache/

Add the following line to ~/.profile

export PIP_CACHE_DIR=$SCRATCH/pip_cache/

pip config set global.cache-dir $SCRATCH/pip_cache
```

# Install packages

```
pip install --upgrade pip

for line in $(cat requirements.txt); do pip install $line; done

cd

cd mq/frame_feature_extractors/ofa

pip install ofa/transformers/

mkdir $SCRATCH/mq_pretrained_models

cd $SCRATCH/mq_pretrained_models

mkdir unidet

gdrive --id 1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq -O unidet/Unified_learned_OCIM_R50_6x+2x.pth

mkdir visor_hos

wget https://www.dropbox.com/s/bfu94fpft2wi5sn/model_final_hos.pth?dl=0 -O visor_hos/model_final_hos.pth

git clone https://huggingface.co/OFA-Sys/OFA-huge

mv OFA-huge ofa

wget https://huggingface.co/OFA-Sys/ofa-huge/resolve/main/pytorch_model.bin -O ofa/pytorch_model.bin

wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_huge.pt -O ofa/ofa_huge.pt

mkdir blip

wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth -O blip/model_base_capfilt_large.pth

python3 -m ipykernel install --user --name=mq
```

# Download dataset

Use the following two commands in the terminal of your own computer:

```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

unzip awscliv2.zip
```

Copy aws folder to somewhere in CVL server, and then run the following commands in the terminal of CVL server.

```
mv aws $SCRATCH/

cd aws

chmod +x install

chmod +x dist/aws

./install -i $SCRATCH/aws/aws-cli -b $SCRATCH/aws/aws-cli-bin

$SCRATCH/aws/aws-cli-bin/aws configure
```

Enter the AWS Access Key ID from the e-mail you received from Ego4D.

Enter the AWS Secret Access Key from the e-mail you received from Ego4D.

Execute the following two commands:
```
screen

ego4d --output_directory="$BEEGFS_SCRATCH/ego4d_data" --datasets full_scale annotations --metadata --benchmarks EM --version v2

rm -rf aws
```
