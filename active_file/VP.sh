export NUM_GPUS="$@"

bash bash/VP/dtd.sh $NUM_GPUS
bash bash/VP/cub200.sh $NUM_GPUS
bash bash/VP/nabirds.sh $NUM_GPUS
bash bash/VP/stanford_dogs.sh $NUM_GPUS
bash bash/VP/flowers102.sh $NUM_GPUS
bash bash/VP/food101.sh $NUM_GPUS
bash bash/VP/cifar100.sh $NUM_GPUS
bash bash/VP/cifar10.sh $NUM_GPUS
bash bash/VP/gtsrb.sh $NUM_GPUS
bash bash/VP/svhn.sh $NUM_GPUS