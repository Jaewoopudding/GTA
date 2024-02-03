datapath=''

mcq_device=cuda:0
cql_device=cuda:1
lqi_device=cuda:2

std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=maze2d-medium-v1
GDA=GTA_tune1
max_timesteps=1000000
batch_size=1024
data_mixture_type=mixed
GDA_id=''

s4rl_augmentation_type=identical
n_episodes=100

lmbda=0.9


python corl/algorithms/mcq.py --n_episodes=$n_episodes --datapath=$datapath --device=$mcq_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
python corl/algorithms/mcq.py --n_episodes=$n_episodes --datapath=$datapath --device=$mcq_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
python corl/algorithms/mcq.py --n_episodes=$n_episodes --datapath=$datapath --device=$mcq_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
python corl/algorithms/mcq.py --n_episodes=$n_episodes --datapath=$datapath --device=$mcq_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
python corl/algorithms/cql.py --n_episodes=$n_episodes --datapath=$datapath --device=$cql_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size &
python corl/algorithms/cql.py --n_episodes=$n_episodes --datapath=$datapath --device=$cql_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size &
python corl/algorithms/cql.py --n_episodes=$n_episodes --datapath=$datapath --device=$cql_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size &
python corl/algorithms/cql.py --n_episodes=$n_episodes --datapath=$datapath --device=$cql_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size &
python corl/algorithms/iql.py --n_episodes=$n_episodes --datapath=$datapath --device=$lqi_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size &
python corl/algorithms/iql.py --n_episodes=$n_episodes --datapath=$datapath --device=$lqi_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size &
python corl/algorithms/iql.py --n_episodes=$n_episodes --datapath=$datapath --device=$lqi_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size &
python corl/algorithms/iql.py --n_episodes=$n_episodes --datapath=$datapath --device=$lqi_device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size &
wait
