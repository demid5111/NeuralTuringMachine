# make the script verbose
set -x

# create a models directory
mkdir models

# run the training
python run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 100000 --steps_per_eval 1000 --use_local_impl yes \
                      --curriculum none \
                      --num_bits_per_vector 3 --num_memory_locations 128 \
                      --max_seq_len 4 --task average_sum \
                      --num_experts 3
