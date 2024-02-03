device="cuda:6"
iteration=2
seed=0

python corl/algorithms/td3_bc.py --env=halfcheetah-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training 
python corl/algorithms/td3_bc.py --env=halfcheetah-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/td3_bc.py --env=halfcheetah-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/td3_bc.py --env=hopper-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/td3_bc.py --env=hopper-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/td3_bc.py --env=hopper-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/td3_bc.py --env=walker2d-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/td3_bc.py --env=walker2d-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/td3_bc.py --env=walker2d-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training

python corl/algorithms/edac.py --env=halfcheetah-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/edac.py --env=halfcheetah-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/edac.py --env=halfcheetah-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/edac.py --env=hopper-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/edac.py --env=hopper-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/edac.py --env=hopper-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/edac.py --env=walker2d-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/edac.py --env=walker2d-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/edac.py --env=walker2d-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training