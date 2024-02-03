device=cuda:3
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=antmaze-umaze-diverse-v2
s4rl_augmentation_type=identical
datapath=/home/orl/jaewoo/v4/antmaze-umaze-diverse-v2/5M-1_1x-smallmixer-50-sar-temp1_0.npz
max_timesteps=1000000
project="augmentation_baselines"
batch_size=1024

# for hopper med, medrep?
tau=0.005
iql_deterministic=False
normalize_reward=True

python corl/algorithms/iql.py --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --max_timesteps=$max_timesteps&
python corl/algorithms/iql.py --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --max_timesteps=$max_timesteps&
python corl/algorithms/iql.py --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --max_timesteps=$max_timesteps&
python corl/algorithms/iql.py --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --max_timesteps=$max_timesteps&
