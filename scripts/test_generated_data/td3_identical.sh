device=cuda:7
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=halfcheetah-medium-v2
GDA=None
s4rl_augmentation_type=identical
batch_size=256
max_timesteps=10000000
project=augmentation_baselines

python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/td3_bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait