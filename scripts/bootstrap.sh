# make the script verbose
set -x

# Accessing the machine:
#
# ssh root@188.166.169.120

# Preparing fresh machine for the first usage:

sudo apt-get --yes update
sudo apt-get --yes upgrade
sudo apt-get --yes install software-properties-common
sudo add-apt-repository --yes ppa:deadsnakes/ppa
sudo apt-get --yes install python3.7
sudo apt-get --yes install python3-pip
python3.7 -m pip install virtualenv
python3.7 -m virtualenv venv
source venv/bin/activate
python -m pip install -r requirements.txt
