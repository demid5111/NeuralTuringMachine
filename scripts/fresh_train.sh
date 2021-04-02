# make the script verbose
set -x

# Watching the training:
#
# tail -f ~/projects/tf1-approved-NeuralTuringMachine/out.log

# Running the training:

pushd ~/NeuralTuringMachine/
source venv/bin/activate
rm -rf models/*
rm -rf out.log
popd

pushd ~/NeuralTuringMachine/
nohup bash scripts/run.sh > out.log &
popd
