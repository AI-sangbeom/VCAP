export NUM_GPUS="$@"

bash bash/FT/dtd.sh $NUM_GPUS
bash bash/FT/cub200.sh $NUM_GPUS
bash bash/FT/nabirds.sh $NUM_GPUS
bash bash/FT/stanford_dogs.sh $NUM_GPUS
bash bash/FT/flowers102.sh $NUM_GPUS
bash bash/FT/food101.sh $NUM_GPUS
bash bash/FT/cifar100.sh $NUM_GPUS
bash bash/FT/cifar10.sh $NUM_GPUS
bash bash/FT/gtsrb.sh $NUM_GPUS
bash bash/FT/svhn.sh $NUM_GPUS