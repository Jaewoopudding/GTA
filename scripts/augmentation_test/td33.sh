env=hopper-medium-replay-v2
device="cuda:7"
iteration=2
replay_ratio=1
seed=0
project=Augmentation

#python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=amplitude_scaling   --group=test
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=state_mix_up  --group=test
python corl/algorithms/td3_bc.py --env=$env --iteration=$iteration --seed=$seed --device=$device --s4rl_augmentation_type=adversarial_state_training  --group=test