
# device=cuda:0
# std_scale=0.0003
# uniform_scale=0.0003
# adv_scale=0.0001
# env=maze2d-large-v1
# GDA_id=5M-1_1x-smallmixer-50-sar-temp1_0
# s4rl_augmentation_type=identical
# datapath=/home/orl/output/output/maze2d-large-v1/smallmixer_denoiser_v4_maze/2024-01-08/04:33/5M-1_25x-smallmixer-50-sar-temp1_5-v6.npz
# batch_size=1024
# max_timesteps=1000000

# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
# python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&



device=cuda:6
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=maze2d-umaze-v1
GDA_id=0119
s4rl_augmentation_type=identical
datapath=results/maze2d-umaze-v1/smallmixer_denoiser_v4_maze/2024-01-19/15:24/5M-1_3x-smallmixer-50-sar-temp2_0.npz
batch_size=1024
max_timesteps=1000000

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
