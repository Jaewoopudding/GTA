# device=cuda:4
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=halfcheetah-medium-v2
# GDA=None
# s4rl_augmentation_type=identical
# datapath=data/generated_data/halfcheetah-medium-v2/smallmixer/5M-1_25x-smallmixer-50-sar.npz
# max_timesteps=1000000
# project="augmentation_baselines_v2"
# batch_size=256

# python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps&
# python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps&
# python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps&
# python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps&
# python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --max_timesteps=$max_timesteps&

device=cuda:1
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=halfcheetah-medium-v2
GDA=synther
s4rl_augmentation_type=identical
max_timesteps=1000000
project="augmentation_baselines_v2"
batch_size=256

python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --max_timesteps=$max_timesteps&
python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --max_timesteps=$max_timesteps&
python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --max_timesteps=$max_timesteps&
python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --max_timesteps=$max_timesteps&
python corl/algorithms/mcq.py --device=$device --project=$project --batch_size=$batch_size --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --datapath=$datapath --max_timesteps=$max_timesteps&
