device="cuda:7"
iteration=2
seed=0

env=hopper-medium-replay-v2

python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=identical
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=gaussian_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=uniform_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=amplitude_scaling 
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=state_mix_up

env=hopper-medium-v2

python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=identical
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=gaussian_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=uniform_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=amplitude_scaling 
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=state_mix_up

env=walker2d-medium-expert-v2

python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=identical
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=gaussian_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=uniform_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=amplitude_scaling 
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=state_mix_up

env=walker2d-medium-replay-v2

python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=identical
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=gaussian_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=uniform_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=amplitude_scaling 
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=state_mix_up

env=walker2d-medium-v2

python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=identical
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=gaussian_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=uniform_noise
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=amplitude_scaling 
python corl/algorithms/edac.py --env=$env --iteration=$iteration --seed=$seed --device=$device --augmentation_type=state_mix_up
