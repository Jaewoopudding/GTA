device=cuda:5
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
envs=halfcheetah-medium-v2
GDA=None
max_timesteps=5000000
batch_size=256

for env in "halfcheetah-medium-v2" "halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2" "halfcheetah-random-v2" "walker2d-medium-v2" "walker2d-medium-expert-v2" "walker2d-medium-replay-v2" "walker2d-random-v2" "hopper-medium-v2" "hopper-medium-expert-v2" "hopper-medium-replay-v2" "hopper-random-v2"  
do
    python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    wait
    python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    wait
done

