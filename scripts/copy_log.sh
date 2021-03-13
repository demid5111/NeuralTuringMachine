# make the script verbose
set -x

WHERE_TO_KEEP=~/Downloads/new_log/
rm -rf ${WHERE_TO_KEEP}
mkdir -p ${WHERE_TO_KEEP}
scp -r root@188.166.169.120:~/projects/tf1-approved-NeuralTuringMachine/out.log ${WHERE_TO_KEEP}
