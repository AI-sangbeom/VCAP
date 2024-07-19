clear

export NUM_GPUS="$@"
CONFIG=configs/config_files/VCAP/cub200.yaml

if [ "$NUM_GPUS" -eq 1 ]
then
    python main.py --cfg $CONFIG
else 
    OMP_NUM_THREADS=$NUM_GPUS torchrun --nproc_per_node=$NUM_GPUS main.py --cfg $CONFIG
fi