export NUM_GPUS="$@"

bash bash/VPT/dtd.sh $NUM_GPUS
bash bash/VPT/cub200.sh $NUM_GPUS
bash bash/VPT/nabirds.sh $NUM_GPUS
bash bash/VPT/stanford_dogs.sh $NUM_GPUS
bash bash/VPT/flowers102.sh $NUM_GPUS
bash bash/VPT/food101.sh $NUM_GPUS
bash bash/VPT/cifar100.sh $NUM_GPUS
bash bash/VPT/cifar10.sh $NUM_GPUS
bash bash/VPT/gtsrb.sh $NUM_GPUS
bash bash/VPT/svhn.sh $NUM_GPUS