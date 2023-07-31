# Add the following lines to ~/.profile

```
export BEEGFS_SCRATCH=/srv/beegfs02/scratch/aarslan_data/data
export SCRATCH=/scratch/$USER
export TMPDIR=/scratch/$USER/pip_temp
export LC_ALL=C.UTF-8
export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf
export PYENV_ROOT="$SCRATCH/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
export PIP_CACHE_DIR=$SCRATCH/pip_cache/
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
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

# Install pyenv

```
chmod +x scripts/install_pyenv.sh

./scripts/install_pyenv.sh

exit
```

# Install packages

Open a new terminal.

```
chmod +x scripts/install_packages.sh

./scripts/install_packages.sh
```

Windows + Shift + P

Python: Clear Cache and Reload Window

# Download pre-trained models

```
chmod +x scripts/download_pretrained_models.sh

./scripts/download_pretrained_models.sh
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
