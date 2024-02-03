sh scripts/GTA_0123/maze2d-umaze-v1/td3_bc.sh &
sh scripts/GTA_0123/maze2d-large-v1/td3_bc.sh &
sh scripts/GTA_0123/maze2d-medium-v1/td3_bc.sh &
wait

sh scripts/GTA_0123/hopper-medium/td3_bc.sh &
sh scripts/GTA_0123/hopper-medium-expert/td3_bc.sh &
sh scripts/GTA_0123/hopper-medium-replay/td3_bc.sh &
sh scripts/GTA_0123/hopper-random/td3_bc.sh &
wait

sh scripts/GTA_0123/walker2d-medium/td3_bc.sh &
sh scripts/GTA_0123/walker2d-medium-expert/td3_bc.sh &
sh scripts/GTA_0123/walker2d-medium-replay/td3_bc.sh &
sh scripts/GTA_0123/walker2d-random/td3_bc.sh &
wait

sh scripts/GTA_0123/halfcheetah-medium/td3_bc.sh &
sh scripts/GTA_0123/halfcheetah-medium-expert/td3_bc.sh &
sh scripts/GTA_0123/halfcheetah-medium-replay/td3_bc.sh &
sh scripts/GTA_0123/halfcheetah-random/td3_bc.sh &
wait 