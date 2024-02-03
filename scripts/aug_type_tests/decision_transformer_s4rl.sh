device="cuda:7"
iteration=2
replay_ratio=1
seed=0
project=augmentation_baselines

# augmentation=gaussian_noise

# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-replay-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-expert-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-replay-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-expert-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-replay-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-expert-v2 

# augmentation=uniform_noise

# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-replay-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-expert-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-replay-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-expert-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-replay-v2 
# python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-expert-v2 

augmentation=amplitude_scaling

python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-replay-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-expert-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-replay-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-expert-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-replay-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-expert-v2 

augmentation=amplitude_scaling_m

python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-replay-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-expert-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-replay-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-expert-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-replay-v2 
python corl/algorithms/dt.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-expert-v2 