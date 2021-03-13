# make the script verbose
set -x

# Accessing the machine:
#
# ssh root@188.166.169.120

# Preparing fresh machine for the first usage:

sudo apt-get update
sudo apt-get upgrade
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.7
sudo apt-get install python3-pip
python3 -m pip install virtualenv
python3 -m virtualenv venv
