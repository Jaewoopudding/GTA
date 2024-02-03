device=cuda:1
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=maze2d-medium-v1
GDA=synther
max_timesteps=1000000
batch_size=1024
frac=0.1
buffer_size=5000000
n_episodes=100

for s4rl_augmentation_type in 'identical'
do
    python corl/algorithms/bc.py --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --frac=$frac --buffer_size=$buffer_size &
    python corl/algorithms/bc.py --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --frac=$frac --buffer_size=$buffer_size &
    python corl/algorithms/bc.py --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --frac=$frac --buffer_size=$buffer_size &
    python corl/algorithms/bc.py --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --frac=$frac --buffer_size=$buffer_size &
    wait
    python corl/algorithms/bc.py --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size --frac=$frac --buffer_size=$buffer_size &
    python corl/algorithms/bc.py --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size --frac=$frac --buffer_size=$buffer_size &
    python corl/algorithms/bc.py --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size --frac=$frac --buffer_size=$buffer_size &
    python corl/algorithms/bc.py --n_episodes=$n_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size --frac=$frac --buffer_size=$buffer_size &
    wait
done

