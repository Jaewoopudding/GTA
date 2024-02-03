sh scripts/S4RL/halfcheetah-medium/bc_10.sh &
sh scripts/S4RL/halfcheetah-medium-expert/bc_10.sh &
sh scripts/S4RL/halfcheetah-medium-replay/bc_10.sh &
sh scripts/S4RL/halfcheetah-random/bc_10.sh &
wait

sh scripts/S4RL/hopper-medium/bc_10.sh &
sh scripts/S4RL/hopper-medium-expert/bc_10.sh &
sh scripts/S4RL/hopper-medium-replay/bc_10.sh &
sh scripts/S4RL/hopper-random/bc_10.sh &
wait

sh scripts/S4RL/walker2d-medium/bc_10.sh &
sh scripts/S4RL/walker2d-medium-expert/bc_10.sh &
sh scripts/S4RL/walker2d-medium-replay/bc_10.sh &
sh scripts/S4RL/walker2d-random/bc_10.sh &
wait

sh scripts/S4RL/maze2d-umaze-v1/bc_10.sh &
sh scripts/S4RL/maze2d-medium-v1/bc_10.sh &
sh scripts/S4RL/maze2d-large-v1/bc_10.sh &
wait
