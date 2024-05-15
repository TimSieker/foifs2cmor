INSTALLATION


Use mamba to create environment (here called foifs2cmor, but can be anything) and install dependencies. See https://docs.esmvaltool.org/en/latest/quickstart/installation.html#installation-of-subpackages.

### install mamba environment
mamba create --name foifs2cmor

conda activate foifs2cmor

### install dependencies

mamba install esmvaltool

python -m pip install 'scitools-iris @ git+https://github.com/TimSieker/iris.git@iris4cmor'

git clone https://github.com/TimSieker/foifs2cmor.git

