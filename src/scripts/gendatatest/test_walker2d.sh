


# device=cuda:6
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=walker2d-medium-expert-v2
# GDA=5M-fixed95-smallmixer-50-sar-temp1_5
# s4rl_augmentation_type=identical
# datapath=results/walker2d-medium-expert-v2/smallmixer_denoiser_v4/2024-01-13/21:30/5M-1_3x-smallmixer-50-sar-temp2_0.npz
# batch_size=1024
# max_timesteps=1000000

# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&



# device=cuda:3
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=walker2d-medium-replay-v2
# GDA=5M-fixed95-smallmixer-50-sar-temp1_5
# s4rl_augmentation_type=identical
# datapath=/home/orl/final_data/walker2d-medium-replay-v2/5M-1_3x-smallmixer-50-sar-temp2_0.npz
# batch_size=1024
# max_timesteps=1000000
# data_mixture_type=mixed

# python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&

device=cuda:6
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=walker2d-medium-expert-v2
GDA=5M-fixed95-smallmixer-50-sar-temp1_5
s4rl_augmentation_type=identical
datapath=results/walker2d-medium-expert-v2/smallmixer_denoiser_v4/2024-01-15/10:08/5M-1_3x-smallmixer-50-sar-temp2_0-rew.npz
batch_size=1024
max_timesteps=1000000
data_mixture_type=mixed

python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --data_mixture_type=$data_mixture_type --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
