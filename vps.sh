# nohup bash vps.sh > out.log &
mkdir models
python run_tasks.py --experiment_name experiment --verbose no \
                      --num_train_steps 100000 --steps_per_eval 1000 --use_local_impl yes \
                      --curriculum none \
                      --num_bits_per_vector 3 --max_seq_len 10 --task sum
