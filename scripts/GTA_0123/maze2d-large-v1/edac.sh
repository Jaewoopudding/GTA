device=cuda:2
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=maze2d-large-v1
GDA=GTA_0117
max_timesteps=1000000
batch_size=1024
eta=0.1
num_critics=10
buffer_size=10000000
eval_episodes=100

for s4rl_augmentation_type in 'identical'
do
    python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --eval_episodes=$eval_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --eval_episodes=$eval_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --eval_episodes=$eval_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --eval_episodes=$eval_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    wait
    # python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --eval_episodes=$eval_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    # python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --eval_episodes=$eval_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    # python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --eval_episodes=$eval_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    # python corl/algorithms/edac.py --eta=$eta --num_critics=$num_critics --eval_episodes=$eval_episodes --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA=$GDA --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size --buffer_size=$buffer_size &
    # wait
done

