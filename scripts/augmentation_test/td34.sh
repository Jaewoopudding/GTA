env=hopper-medium-replay-v2
device="cuda:7"
iteration=2
replay_ratio=1
seed=0

#python corl/algorithms/td3_bc.py --env=$env --iteration=$replay_ratio --seed=$seed --device=$device --s4rl_augmentation_type=identical --batch_size=512  --group=test
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=gaussian_noise  --group=test
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=uniform_noise  --group=test
