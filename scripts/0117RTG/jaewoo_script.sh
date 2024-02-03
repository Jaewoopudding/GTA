# device=cuda:0
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=halfcheetah-medium-expert-v2
GDA=GTA
# max_timesteps=1000000
# batch_size=1024
# data_mixture_type=''
# GDA_id='genonly'

# for s4rl_augmentation_type in 'identical'
# do
#     python corl/algorithms/td3_bc.py --GDA_id=$GDA_id --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size &
#     python corl/algorithms/td3_bc.py --GDA_id=$GDA_id --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size &
#     python corl/algorithms/td3_bc.py --GDA_id=$GDA_id --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size &
#     python corl/algorithms/td3_bc.py --GDA_id=$GDA_id --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size &
# done

# sleep 5

# device=cuda:1
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=hopper-medium-replay-v2
GDA=GTA
# max_timesteps=1000000
# batch_size=1024

# tau=0.001
# iql_deterministic=True
# normalize_reward=True

# for s4rl_augmentation_type in 'identical'
# do
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
# done

# sleep 5
# device=cuda:2
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=hopper-medium-replay-v2
GDA=GTA
# max_timesteps=1000000
# batch_size=1024
# data_mixture_type=mixed


# for s4rl_augmentation_type in 'identical'
# do
#     python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size &
#     python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size &
#     python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size &
#     python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size &
# done
# sleep 5

device=cuda:3
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=hopper-medium-replay-v2
GDA=GTA
max_timesteps=1000000
batch_size=1024
data_mixture_type=''
GDA_id='genonly'

for s4rl_augmentation_type in 'identical'
do
    python corl/algorithms/td3_bc.py --GDA_id=$GDA_id --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/td3_bc.py --GDA_id=$GDA_id --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/td3_bc.py --GDA_id=$GDA_id --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/td3_bc.py --GDA_id=$GDA_id --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size &
done
sleep 5


# device=cuda:4
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=hopper-medium-v2
GDA=GTA
# max_timesteps=1000000
# batch_size=1024
# # Hyperparameter difference 
# tau=0.001
# iql_deterministic=True
# normalize_reward=True

# for s4rl_augmentation_type in 'identical'
# do
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
# done
# sleep 5

# device=cuda:5
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=hopper-medium-v2
GDA=GTA
# max_timesteps=1000000
# batch_size=1024

# # Hyperparameter difference 
# tau=0.001
# iql_deterministic=True
# normalize_reward=True

# for s4rl_augmentation_type in 'identical'
# do
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
#     python corl/algorithms/iql.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --tau=$tau --iql_deterministic=$iql_deterministic --normalize_reward=$normalize_reward &
# done
