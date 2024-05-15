INSTALLATION


Use mamba to create environment (here called foifs2cmor, but can be anything) and install dependencies. See https://docs.esmvaltool.org/en/latest/quickstart/installation.html#installation-of-subpackages.

### install mamba environment
mamba create --name foifs2cmor

conda activate foifs2cmor

### install dependencies

mamba install esmvaltool=2.10.0

python -m pip install 'scitools-iris @ git+https://github.com/TimSieker/iris.git@iris4cmor'

git clone https://github.com/TimSieker/foifs2cmor.git


### run the code

Adjust all the parameters in the configuration files

then run:

python foci_atmos.py config-user_atmos.yml 

# user configuration
used set up directories and configuration for running the script

# cmor configuration
defines cmor specific parameters
