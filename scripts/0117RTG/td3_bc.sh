sh scripts/0117RTG/maze2d-umaze-v1/td3_bc.sh &
sh scripts/0117RTG/maze2d-large-v1/td3_bc.sh &
sh scripts/0117RTG/maze2d-medium-v1/td3_bc.sh &
wait

sh scripts/0117RTG/hopper-medium/td3_bc.sh &
sh scripts/0117RTG/hopper-medium-expert/td3_bc.sh &
sh scripts/0117RTG/hopper-medium-replay/td3_bc.sh &
sh scripts/0117RTG/hopper-random/td3_bc.sh &
wait

sh scripts/0117RTG/walker2d-medium/td3_bc.sh &
sh scripts/0117RTG/walker2d-medium-expert/td3_bc.sh &
sh scripts/0117RTG/walker2d-medium-replay/td3_bc.sh &
sh scripts/0117RTG/walker2d-random/td3_bc.sh &
wait

sh scripts/0117RTG/halfcheetah-medium/td3_bc.sh &
sh scripts/0117RTG/halfcheetah-medium-expert/td3_bc.sh &
sh scripts/0117RTG/halfcheetah-medium-replay/td3_bc.sh &
sh scripts/0117RTG/halfcheetah-random/td3_bc.sh &
wait 