device=cuda:6
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=halfcheetah-medium-v2
s4rl_augmentation_type=identical
datapath=data/generated_data/v5/halfcheetah-medium-v2/5M-1_25x-smallmixer-50-sar-temp2_0.npz
batch_size=1024
max_timesteps=1000000
project=augmentation_baselines_v2
data_mixture_type=mixed

python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --project=$project --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --project=$project --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --project=$project --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --project=$project --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --project=$project --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&

wait