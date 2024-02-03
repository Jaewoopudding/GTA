device=cuda:6
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=maze2d-medium-v1
GDA=GTA_0126
max_timesteps=1000000
batch_size=1024
buffer_size=10000000
n_episodes=100
alpha=2.5
datapath=results/maze2d-medium-v1/smallmixer_denoiser_v4_maze/2024-01-26/03:51/5M-1_1x-smallmixer-50-sar-temp2_0_10.npz
data_mixture_type="mixed"

for s4rl_augmentation_type in 'identical'
do
    python corl/algorithms/td3_bc.py --datapath=$datapath --data_mixture_type=$data_mixture_type --alpha=$alpha --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    python corl/algorithms/td3_bc.py --datapath=$datapath --data_mixture_type=$data_mixture_type --alpha=$alpha --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    python corl/algorithms/td3_bc.py --datapath=$datapath --data_mixture_type=$data_mixture_type --alpha=$alpha --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    python corl/algorithms/td3_bc.py --datapath=$datapath --data_mixture_type=$data_mixture_type --alpha=$alpha --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    wait
    # python corl/algorithms/td3_bc.py --datapath=$datapath --data_mixture_type=$data_mixture_type --alpha=$alpha --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    # python corl/algorithms/td3_bc.py --datapath=$datapath --data_mixture_type=$data_mixture_type --alpha=$alpha --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    # python corl/algorithms/td3_bc.py --datapath=$datapath --data_mixture_type=$data_mixture_type --alpha=$alpha --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    # python corl/algorithms/td3_bc.py --datapath=$datapath --data_mixture_type=$data_mixture_type --alpha=$alpha --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    # wait
done

