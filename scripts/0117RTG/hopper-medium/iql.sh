
device=cuda:0
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=hopper-medium-v2
GDA=GTA
max_timesteps=1000000
batch_size=1024
datapath=results/hopper-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-16/22:20/5M-1_3x-smallmixer-50-sar-temp2_0.npz
data_mixture_type=mixed
# Hyperparameter difference 
tau=0.001
iql_deterministic=True
normalize_reward=True

for s4rl_augmentation_type in 'identical'
do
    python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
    python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
    python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
    python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
    wait
    # python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
    # python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
    # python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
    # python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
    # wait
done
