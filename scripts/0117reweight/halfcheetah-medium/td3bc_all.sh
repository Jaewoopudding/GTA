

device1=6
device2=7
device3=0
device3=3
data_mixture_type1="mixed"
data_mixture_type2=""

ext="temp"

p1=results/halfcheetah-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-26/13:54/5M-1_3x-smallmixer-100-sar-temp2_0_50.npz
p2=results/halfcheetah-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-26/06:48/5M-1_4x-smallmixer-50-sar-temp2_0_50.npz
p3=results/halfcheetah-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-26/06:45/5M-1_1x-smallmixer-50-sar-temp2_0_50.npz
p4=results/halfcheetah-medium-v2/smallmixer_denoiser_v4_reweighting/2024-01-26/05:00/5M-1_2x-smallmixer-50-sar-temp2_0_50.npz



# bash scripts/0117reweight/halfcheetah-medium/td3_bc.sh $device1 $p1 $data_mixture_type1 &\
# bash scripts/0117reweight/halfcheetah-medium/td3_bc.sh $device2 $p2 $data_mixture_type1 &
bash scripts/0117reweight/halfcheetah-medium/td3_bc.sh $device3 $p3 $data_mixture_type1 &\
bash scripts/0117reweight/halfcheetah-medium/td3_bc.sh $device4 $p4 $data_mixture_type1 &\