device=cuda:1
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=halfcheetah-medium-v2
GDA=5M-1_1x-temporalattention-50-sar
datapath=/home/jaewoo/practices/diffusers/Augmentation-For-OfflineRL/data/generated_data/5M-1_1x-temporalattention-50-sar.npz
batch_size=256
max_timesteps=1000000

python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --datapath=$datapath&
python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --datapath=$datapath&
python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --datapath=$datapath&
python corl/algorithms/sac.py --device=$device --env=$env --GDA=$GDA --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --datapath=$datapath&