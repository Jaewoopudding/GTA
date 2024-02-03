device=cuda:4
std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=walker2d-medium-v2
GDA_id=GTA_reweight_0124
max_timesteps=1000000
batch_size=1024
lmbda=0.9
datapath=results/walker2d-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-24/20:43/5M-1_3x-smallmixer-50-sar-temp2_0.npz
data_mixture_type="mixed"

for s4rl_augmentation_type in 'identical'
do
    python corl/algorithms/mcq.py --device=$device --datapath=$datapath --data_mixture_type=$data_mixture_type --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
    python corl/algorithms/mcq.py --device=$device --datapath=$datapath --data_mixture_type=$data_mixture_type --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
    python corl/algorithms/mcq.py --device=$device --datapath=$datapath --data_mixture_type=$data_mixture_type --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
    python corl/algorithms/mcq.py --device=$device --datapath=$datapath --data_mixture_type=$data_mixture_type --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
    wait
    # python corl/algorithms/mcq.py --device=$device --datapath=$datapath --data_mixture_type=$data_mixture_type --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
    # python corl/algorithms/mcq.py --device=$device --datapath=$datapath --data_mixture_type=$data_mixture_type --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
    # python corl/algorithms/mcq.py --device=$device --datapath=$datapath --data_mixture_type=$data_mixture_type --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
    # python corl/algorithms/mcq.py --device=$device --datapath=$datapath --data_mixture_type=$data_mixture_type --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --GDA_id=$GDA_id --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size --lmbda=$lmbda &
    # wait
done

