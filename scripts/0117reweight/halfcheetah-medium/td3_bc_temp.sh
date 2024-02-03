
device=cuda:3
datapath=results/halfcheetah-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-26/13:54/5M-1_3x-smallmixer-100-sar-temp2_0_50.npz

echo $device
echo $datapath

std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=halfcheetah-medium-v2
max_timesteps=1000000
batch_size=1024
GDA_id=GTA_Ablations_reweighting
data_mixture_type="mixed"

for s4rl_augmentation_type in 'identical'
do  
    echo $GDA_id
    python corl/algorithms/td3_bc.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/td3_bc.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/td3_bc.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/td3_bc.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    wait
    # python corl/algorithms/td3_bc.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    # python corl/algorithms/td3_bc.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    # python corl/algorithms/td3_bc.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    # python corl/algorithms/td3_bc.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    # wait
done

wait