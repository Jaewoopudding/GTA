device=cuda:0
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=walker2d-medium-v2
GDA=None
max_timesteps=1000000
batch_size=1024
rollout_length=1
cql_weight=5.0

for s4rl_augmentation_type in 'identical' 'gaussian_noise'
do
    python corl/algorithms/combo.py --rollout_length=$rollout_length --cql_weight=$cql_weight --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/combo.py --rollout_length=$rollout_length --cql_weight=$cql_weight --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/combo.py --rollout_length=$rollout_length --cql_weight=$cql_weight --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/combo.py --rollout_length=$rollout_length --cql_weight=$cql_weight --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    wait
    python corl/algorithms/combo.py --rollout_length=$rollout_length --cql_weight=$cql_weight --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/combo.py --rollout_length=$rollout_length --cql_weight=$cql_weight --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/combo.py --rollout_length=$rollout_length --cql_weight=$cql_weight --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/combo.py --rollout_length=$rollout_length --cql_weight=$cql_weight --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    wait
done

