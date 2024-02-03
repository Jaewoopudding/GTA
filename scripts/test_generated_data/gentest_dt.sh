device=cuda:4
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=hopper-medium-v2
GDA=5M-1_25x-smallmixer-50-sar
s4rl_augmentation_type=identical
datapath=/home/orl/jaewoo/v2/halfcheetah-medium-v2/5M-1_15x-smallmixer-50-sar-temp0_1.npz
batch_size=64
project="augmentation_baselines_v2"

python corl/algorithms/dt.py --device=$device --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --batch_size=$batch_size&
python corl/algorithms/dt.py --device=$device --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --batch_size=$batch_size&
python corl/algorithms/dt.py --device=$device --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --batch_size=$batch_size&
python corl/algorithms/dt.py --device=$device --project=$project --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --batch_size=$batch_size&
