# GTA: Generative Trajectory Augmentation with Guidance for Offline Reinforcement Learning

### Install Dependencies
To install dependecies, please run the command `pip install -r requirement.txt`.

### Code references
Our implementation is heavily based on "Synthetic Experience Replay" (https://github.com/conglu1997/SynthER). 

### Train Diffusion Model
To train diffusion model, please run the following command
```
    python src/diffusion/train_diffuser.py --dataset "<env_name>-<dataset_type>-v2" --config_name <config_name>
```

### Augment Trajectory-Level Data
To sample augmented data from trained diffusion model, please run the following command
```
    python src/diffusion/train_diffuser.py --dataset "<env_name>-<dataset_type>-v2" --config_name <config_name> --load_checkpoint --ckpt_path <ckpt_path> --back_and_forth
```

### Train Offline RL Algorithm
To train offline RL algorithms with augmented dataset, please run the following command
```
    python corl/algorithms/td3bc.py --dataset "<env_name>-<dataset_type>-v2" --GDA GTA --seed 0 --max_timesteps 1000000 --batch_size 1024
```
