rm -rf $SCRATCH/.pyenv

rm -rf $SCRATCH/pip_cache

rm -rf $SCRATCH/pip_temp

cd ~/mq

chmod +x scripts/pyenv-installer

bash ./scripts/pyenv-installer.sh

env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.9

pyenv rehash

pyenv global 3.9.9

pyenv update