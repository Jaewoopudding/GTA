device="cuda:6"
iteration=2
replay_ratio=1
seed=0
augmentation=state_mix_up
project=augmentation_baselines


# python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-v2 
# python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-replay-v2 
# python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-expert-v2 

python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-v2 
python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-replay-v2 
python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-expert-v2 

# python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-v2 
# python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-replay-v2 
# python corl/algorithms/td3_bc.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-expert-v2 


# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-v2 
# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-replay-v2 
# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-expert-v2 

# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-v2 
# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-replay-v2 
# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-expert-v2 

# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-v2 
# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-replay-v2 
# python corl/algorithms/cql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-expert-v2 


# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-v2 
# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-replay-v2 
# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-expert-v2 

# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-v2 
# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-replay-v2 
# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-expert-v2 

# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-v2 
# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-replay-v2 
# python corl/algorithms/iql.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-expert-v2 


# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-v2 
# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-replay-v2 
# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=halfcheetah-medium-expert-v2 

# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-v2 
# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-replay-v2 
# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=hopper-medium-expert-v2 

# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-v2 
# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-replay-v2 
# python corl/algorithms/edac.py --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=$augmentation --project=$project --env=walker2d-medium-expert-v2 