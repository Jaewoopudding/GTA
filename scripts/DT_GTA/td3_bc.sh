sh scripts/S4RL/halfcheetah-medium/td3_bc.sh &
sh scripts/S4RL/halfcheetah-medium-expert/td3_bc.sh &
sh scripts/S4RL/halfcheetah-medium-replay/td3_bc.sh &
sh scripts/S4RL/halfcheetah-random/td3_bc.sh &
wait

sh scripts/S4RL/hopper-medium/td3_bc.sh &
sh scripts/S4RL/hopper-medium-expert/td3_bc.sh &
sh scripts/S4RL/hopper-medium-replay/td3_bc.sh &
sh scripts/S4RL/hopper-random/td3_bc.sh &
wait

sh scripts/S4RL/walker2d-medium/td3_bc.sh &
sh scripts/S4RL/walker2d-medium-expert/td3_bc.sh &
sh scripts/S4RL/walker2d-medium-replay/td3_bc.sh &
sh scripts/S4RL/walker2d-random/td3_bc.sh &
wait

sh scripts/S4RL/maze2d-umaze-v1/td3_bc.sh &
sh scripts/S4RL/maze2d-medium-v1/td3_bc.sh &
sh scripts/S4RL/maze2d-large-v1/td3_bc.sh &
wait
