export NUM_GPUS="$@"

bash bash/HT/dtd.sh $NUM_GPUS
bash bash/HT/cub200.sh $NUM_GPUS
bash bash/HT/nabirds.sh $NUM_GPUS
bash bash/HT/stanford_dogs.sh $NUM_GPUS
bash bash/HT/flowers102.sh $NUM_GPUS
bash bash/HT/food101.sh $NUM_GPUS
bash bash/HT/cifar100.sh $NUM_GPUS
bash bash/HT/cifar10.sh $NUM_GPUS
bash bash/HT/gtsrb.sh $NUM_GPUS
bash bash/HT/svhn.sh $NUM_GPUS