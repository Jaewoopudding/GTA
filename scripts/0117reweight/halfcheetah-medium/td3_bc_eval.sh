
# device=$1
# datapath=$2

# echo $device
# echo $datapath


# done : s4rl GTA
# todo :, SER

device=cuda:1
datapath=/home/orl/ablations-halfcheetah-medium-v2/envtest/halfcheetah-medium-v2-ser.npy

std_scale=0.0003
uniform_scale=0.0003
adv_scale=0.0001
env=halfcheetah-medium-v2
max_timesteps=1000000
batch_size=1024
GDA_id=GTA_Ablations_forbuffer2
data_mixture_type=""
for s4rl_augmentation_type in 'identical'
do  
    echo $GDA_id
    python corl/algorithms/td3_bc_eval.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=0 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/td3_bc_eval.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=1 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/td3_bc_eval.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=2 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    python corl/algorithms/td3_bc_eval.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=3 --max_timesteps=$max_timesteps --batch_size=$batch_size&
    wait
    # python corl/algorithms/td3_bc_evaluation.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=4 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    # python corl/algorithms/td3_bc_evaluation.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=5 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    # python corl/algorithms/td3_bc_evaluation.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=6 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    # python corl/algorithms/td3_bc_evaluation.py --data_mixture_type=$data_mixture_type --datapath=$datapath --GDA_id=$GDA_id --device=$device --s4rl_augmentation_type=$s4rl_augmentation_type --adv_scale=$adv_scale --std_scale=$std_scale --uniform_scale=$uniform_scale --env=$env --seed=7 --max_timesteps=$max_timesteps --batch_size=$batch_size &
    # wait
done

wait