# Accessing the machine:
#
# ssh root@188.166.169.120

# Preparing fresh machine for the first usage:
#
# sudo apt-get update
# sudo apt-get upgrade
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt-get install python3.7
# sudo apt-get install python3-pip
# python3 -m pip install virtualenv
# python3 -m virtualenv -p `which python3.7` venv

# Running the training:
#
# cd projects/tf1-approved-NeuralTuringMachine/
# source venv/bin/activate
# rm -rf models/*
# rm -rf out.log
# nohup bash vps.sh > out.log &

# Watching the training:
#
# tail -f ~/projects/tf1-approved-NeuralTuringMachine/out.log

# Copy the trained model:
#
# scp -r root@188.166.169.120:~/projects/tf1-approved-NeuralTuringMachine/models/115000 ~/Downloads/model_tf2

mkdir models
python run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 100000 --steps_per_eval 1000 --use_local_impl yes \
                      --curriculum none \
                      --num_bits_per_vector 3 --max_seq_len 10 --task sum
