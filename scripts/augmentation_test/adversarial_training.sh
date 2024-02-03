device="cuda:6"
iteration=2
seed=0


python corl/algorithms/cql.py --env=halfcheetah-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/cql.py --env=halfcheetah-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/cql.py --env=halfcheetah-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/cql.py --env=hopper-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/cql.py --env=hopper-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/cql.py --env=hopper-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/cql.py --env=walker2d-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/cql.py --env=walker2d-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/cql.py --env=walker2d-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training

python corl/algorithms/iql.py --env=halfcheetah-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/iql.py --env=halfcheetah-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/iql.py --env=halfcheetah-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/iql.py --env=hopper-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/iql.py --env=hopper-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/iql.py --env=hopper-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/iql.py --env=walker2d-medium-expert-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/iql.py --env=walker2d-medium-replay-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
python corl/algorithms/iql.py --env=walker2d-medium-v2 --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training
