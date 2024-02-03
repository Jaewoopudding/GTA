

tar -xvf /data/maze2d-medium-v1.tar.gz -C /data


start_id=0
gpu_id=$start_id
total_gpu=3

env="maze2d-medium-v1"
s4rl_augmentation_type='identical'
path=/data/jaewoo/v6/maze2d-medium-v1
batch_size=1024
diffuser_architecture=smallmixer
alpha=2.5


for datavolume in 5 
do
    for guidance_rewardscale in 1.25 1.3 1.2 1.10 1.15 1.35
    do
        for noise_level in 50
        do  
            for temperature in 1.0 1.5 2.0
            do
                datapath="${path}/${datavolume}M-${guidance_rewardscale/./_}x-${diffuser_architecture}-${noise_level}-sar-temp${temperature/./_}.npz"
                if [ $gpu_id -eq $total_gpu ]; then
                    for seed in 0 1 2 3 4
                    do
                        device="cuda:$gpu_id"
                        python corl/algorithms/generated_data_validation.py --device=$device --alpha=$alpha --s4rl_augmentation_type=$s4rl_augmentation_type --env=$env --seed=$seed --datapath=$datapath --batch_size=$batch_size&
                    done
                    gpu_id=$start_id
                    wait
                else
                    for seed in 0 1 2 3 4
                    do
                        device="cuda:$gpu_id"
                        python corl/algorithms/generated_data_validation.py --device=$device --alpha=$alpha --s4rl_augmentation_type=$s4rl_augmentation_type --env=$env --seed=$seed --datapath=$datapath --batch_size=$batch_size&
                    done
                    gpu_id=$((gpu_id+1))
                fi
            done
        done
    done
done

wait
