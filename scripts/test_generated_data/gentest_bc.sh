device=cuda:5
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=hopper-medium-v2
GDA=5M-1_25x-mixer-50-sar-temp1.0
s4rl_augmentation_type=identical
datapath=/home/orl/jaewoo/v2/hopper-medium-v2/5M-uncond-smallmixer-50-sar-temp1_0.npz
max_timesteps=1000000
batch_size=1024
frac=1.0

python corl/algorithms/bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps --frac=$frac&
python corl/algorithms/bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps --frac=$frac&
python corl/algorithms/bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps --frac=$frac&
python corl/algorithms/bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps --frac=$frac&
python corl/algorithms/bc.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps --frac=$frac&
