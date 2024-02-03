env=hopper-medium-v2
device="cuda:5"
iteration=2
replay_ratio=1
seed=0

python corl/algorithms/cql.py --env=$env --iteration=$replay_ratio --seed=$seed --device=$device --augmentation_type=identical
python corl/algorithms/cql.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=gaussian_noise 
python corl/algorithms/cql.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=uniform_noise
python corl/algorithms/cql.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=amplitude_scaling 
python corl/algorithms/cql.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=state_mix_up
python corl/algorithms/cql.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training

python corl/algorithms/td3_bc.py --env=$env --iteration=$replay_ratio --seed=$seed --device=$device --augmentation_type=identical
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=gaussian_noise
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=uniform_noise
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=amplitude_scaling 
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=state_mix_up
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training

python corl/algorithms/edac.py --env=$env --iteration=$replay_ratio --seed=$seed --device=$device --augmentation_type=identical
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=gaussian_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=uniform_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=amplitude_scaling 
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=state_mix_up
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=adversarial_state_training