
start_id=6
gpu_id=$start_id
total_gpu=7

env="hopper-medium-v2"
s4rl_augmentation_type='identical'
path=data/generated_data/v5/hopper-medium-v2
batch_size=1024
diffuser_architecture=smallmixer


for datavolume in 5 
do
    for guidance_rewardscale in 1.05 1.1 1.15 1.2 1.25 1.3
    do
        for noise_level in 25 50
        do  
            for temperature in 0.5 1.0 2.0
            do
                datapath="${path}/${datavolume}M-${guidance_rewardscale/./_}x-${diffuser_architecture}-${noise_level}-sar-temp${temperature/./_}.npz"
                if [ $gpu_id -eq $total_gpu ]; then
                    for seed in 0 1 2 3 4
                    do
                        device="cuda:$gpu_id"
                        python corl/algorithms/bc4gen.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --env=$env --seed=$seed --datapath=$datapath --batch_size=$batch_size&
                    done
                    gpu_id=$start_id
                    wait
                else
                    for seed in 0 1 2 3 4
                    do
                        device="cuda:$gpu_id"
                        python corl/algorithms/bc4gen.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --env=$env --seed=$seed --datapath=$datapath --batch_size=$batch_size&
                    done
                    gpu_id=$((gpu_id+1))
                fi
            done
        done
    done
done

wait
