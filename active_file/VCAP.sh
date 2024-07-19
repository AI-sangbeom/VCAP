export NUM_GPUS="$@"

bash bash/VCAP/dtd.sh $NUM_GPUS
bash bash/VCAP/cub200.sh $NUM_GPUS
bash bash/VCAP/nabirds.sh $NUM_GPUS
bash bash/VCAP/stanford_dogs.sh $NUM_GPUS
bash bash/VCAP/flowers102.sh $NUM_GPUS
bash bash/VCAP/food101.sh $NUM_GPUS
bash bash/VCAP/cifar100.sh $NUM_GPUS
bash bash/VCAP/cifar10.sh $NUM_GPUS
bash bash/VCAP/gtsrb.sh $NUM_GPUS
bash bash/VCAP/svhn.sh $NUM_GPUS