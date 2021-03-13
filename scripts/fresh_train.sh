# make the script verbose
set -x

# Watching the training:
#
# tail -f ~/projects/tf1-approved-NeuralTuringMachine/out.log

# Running the training:

pushd ~/projects/tf1-approved-NeuralTuringMachine/
source venv/bin/activate
rm -rf models/*
rm -rf out.log
popd

pushd ~/projects/tf1-approved-NeuralTuringMachine/
nohup bash scripts/run.sh > out.log &
popd
