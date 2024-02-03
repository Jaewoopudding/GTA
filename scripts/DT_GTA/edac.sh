sh scripts/S4RL/halfcheetah-medium/edac.sh &
sh scripts/S4RL/halfcheetah-medium-expert/edac.sh &
sh scripts/S4RL/halfcheetah-medium-replay/edac.sh &
sh scripts/S4RL/halfcheetah-random/edac.sh &
wait

sh scripts/S4RL/hopper-medium/edac.sh &
sh scripts/S4RL/hopper-medium-expert/edac.sh &
sh scripts/S4RL/hopper-medium-replay/edac.sh &
sh scripts/S4RL/hopper-random/edac.sh &
wait

sh scripts/S4RL/walker2d-medium/edac.sh &
sh scripts/S4RL/walker2d-medium-expert/edac.sh &
sh scripts/S4RL/walker2d-medium-replay/edac.sh &
sh scripts/S4RL/walker2d-random/edac.sh &
wait

sh scripts/S4RL/maze2d-umaze-v1/edac.sh &
sh scripts/S4RL/maze2d-medium-v1/edac.sh &
sh scripts/S4RL/maze2d-large-v1/edac.sh &
wait
