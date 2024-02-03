sh scripts/S4RL/halfcheetah-medium/mcq.sh &
sh scripts/S4RL/halfcheetah-medium-expert/mcq.sh &
sh scripts/S4RL/halfcheetah-medium-replay/mcq.sh &
sh scripts/S4RL/halfcheetah-random/mcq.sh &
wait

sh scripts/S4RL/hopper-medium/mcq.sh &
sh scripts/S4RL/hopper-medium-expert/mcq.sh &
sh scripts/S4RL/hopper-medium-replay/mcq.sh &
sh scripts/S4RL/hopper-random/mcq.sh &
wait

sh scripts/S4RL/walker2d-medium/mcq.sh &
sh scripts/S4RL/walker2d-medium-expert/mcq.sh &
sh scripts/S4RL/walker2d-medium-replay/mcq.sh &
sh scripts/S4RL/walker2d-random/mcq.sh &
wait

sh scripts/S4RL/maze2d-umaze-v1/mcq.sh &
sh scripts/S4RL/maze2d-medium-v1/mcq.sh &
sh scripts/S4RL/maze2d-large-v1/mcq.sh &
wait
