device=cuda:4
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=hopper-medium-replay-v2
GDA=None
s4rl_augmentation_type=identical
batch_size=1024
max_timesteps=1000000
project=iql_debugging
group=hyperparameter_tuned

tau=0.005   # Target network update rate ## TODO
iql_deterministic=False  # Use deterministic actor ## TODO
normalize_reward=False    # Normalize reward ## TODO

python corl/algorithms/iql.py --device=$device --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --batch_size=$batch_size --max_timesteps=$max_timesteps --group=$group&
python corl/algorithms/iql.py --device=$device --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --batch_size=$batch_size --max_timesteps=$max_timesteps --group=$group&
python corl/algorithms/iql.py --device=$device --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --batch_size=$batch_size --max_timesteps=$max_timesteps --group=$group&
python corl/algorithms/iql.py --device=$device --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --batch_size=$batch_size --max_timesteps=$max_timesteps --group=$group&
python corl/algorithms/iql.py --device=$device --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --batch_size=$batch_size --max_timesteps=$max_timesteps --group=$group&
wait