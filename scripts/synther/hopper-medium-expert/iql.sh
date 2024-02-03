std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=hopper-medium-expert-v2
GDA=synther
max_timesteps=1000000
batch_size=1024
beta=6.0
iql_tau=0.5
data_mixture_type=''

for s4rl_augmentation_type in 'identical'
do
    device=cuda:0
    python corl/algorithms/iql.py --data_mixture_type=$data_mixture_type --beta=$beta --iql_tau=$iql_tau --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/iql.py --data_mixture_type=$data_mixture_type --beta=$beta --iql_tau=$iql_tau --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/iql.py --data_mixture_type=$data_mixture_type --beta=$beta --iql_tau=$iql_tau --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/iql.py --data_mixture_type=$data_mixture_type --beta=$beta --iql_tau=$iql_tau --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    device=cuda:1
    python corl/algorithms/iql.py --data_mixture_type=$data_mixture_type --beta=$beta --iql_tau=$iql_tau --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/iql.py --data_mixture_type=$data_mixture_type --beta=$beta --iql_tau=$iql_tau --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/iql.py --data_mixture_type=$data_mixture_type --beta=$beta --iql_tau=$iql_tau --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/iql.py --data_mixture_type=$data_mixture_type --beta=$beta --iql_tau=$iql_tau --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    wait
done
