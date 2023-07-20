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
srun --time 720 --gres=gpu:4 --nodelist=biwirender07 --pty bash -i

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

Check if the pip cache location is set correctly

pip cache info
```

# Install packages

```
pip install --upgrade pip

cd

cd mq

for line in $(cat requirements.txt); do pip install $line; done

mkdir $SCRATCH/mq_libs

cd $SCRATCH/mq_libs

git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git

pip install OFA/transformers/

python3 -m ipykernel install --user --name=mq

git clone https://huggingface.co/OFA-Sys/OFA-huge

wget https://huggingface.co/OFA-Sys/ofa-huge/resolve/main/pytorch_model.bin -O $SCRATCH/mq_libs/OFA-huge/pytorch_model.bin

wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_huge.pt -O $SCRATCH/mq_libs/OFA-huge/ofa_huge.pt
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
