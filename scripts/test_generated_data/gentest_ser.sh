device=cuda:5
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=halfcheetah-medium-v2
GDA=synther
s4rl_augmentation_type=identical
batch_size=1024
max_timesteps=1000000

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait

env=halfcheetah-random-v2

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait

env=halfcheetah-medium-expert-v2

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait

env=halfcheetah-medium-replay-v2

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait

python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
python corl/algorithms/generated_data_validation.py --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --datapath=$datapath --batch_size=$batch_size --max_timesteps=$max_timesteps&
wait
