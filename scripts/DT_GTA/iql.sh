sh scripts/S4RL/halfcheetah-medium/iql.sh &
sh scripts/S4RL/halfcheetah-medium-expert/iql.sh &
sh scripts/S4RL/halfcheetah-medium-replay/iql.sh &
sh scripts/S4RL/halfcheetah-random/iql.sh &
wait

sh scripts/S4RL/hopper-medium/iql.sh &
sh scripts/S4RL/hopper-medium-expert/iql.sh &
sh scripts/S4RL/hopper-medium-replay/iql.sh &
sh scripts/S4RL/hopper-random/iql.sh &
wait

sh scripts/S4RL/walker2d-medium/iql.sh &
sh scripts/S4RL/walker2d-medium-expert/iql.sh &
sh scripts/S4RL/walker2d-medium-replay/iql.sh &
sh scripts/S4RL/walker2d-random/iql.sh &
wait

sh scripts/S4RL/maze2d-umaze-v1/iql.sh &
sh scripts/S4RL/maze2d-medium-v1/iql.sh &
sh scripts/S4RL/maze2d-large-v1/iql.sh &
wait
