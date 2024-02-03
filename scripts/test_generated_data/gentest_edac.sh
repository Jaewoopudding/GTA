device=cuda:7
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=halfcheetah-medium-expert-v2
GDA=None
s4rl_augmentation_type=identical
#datapath=/home/jaewoo/practices/diffusers/Augmentation-For-OfflineRL/data/generated_data/halfcheetah-medium-v2/smallmixer/5M-1_25x-smallmixer-50-sar.npz
batch_size=1024
max_timesteps=1000000
buffer_size=2000000
eta=5.0
num_critics=10

for s4rl_augmentation_type in 'identical' 'gaussian_noise'
do
    for seed in 0 1 2 3 4 5 6 7 
    do
        python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=$seed --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps --buffer_size=$buffer_size&
        if [ $seed -eq 3 ]; then
            wait
        fi
    done
    wait

    env=halfcheetah-medium-expert-v2
    eta=0.0
    num_critics=10

    for seed in 0 1 2 3 4 5 6 7 
    do
        python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=$seed --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps --buffer_size=$buffer_size&
        if [ $seed -eq 3 ]; then
            wait
        fi
    done
done