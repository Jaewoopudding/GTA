sh scripts/S4RL/halfcheetah-medium/dt.sh &
sleep 120
sh scripts/S4RL/halfcheetah-medium-expert/dt.sh &
sleep 120
sh scripts/S4RL/halfcheetah-medium-replay/dt.sh &
sleep 120
sh scripts/S4RL/halfcheetah-random/dt.sh &
wait

sh scripts/S4RL/hopper-medium/dt.sh &
sleep 120
sh scripts/S4RL/hopper-medium-expert/dt.sh &
sleep 120
sh scripts/S4RL/hopper-medium-replay/dt.sh &
sleep 120
sh scripts/S4RL/hopper-random/dt.sh &
wait

sh scripts/S4RL/walker2d-medium/dt.sh &
sleep 120
sh scripts/S4RL/walker2d-medium-expert/dt.sh &
sleep 120
sh scripts/S4RL/walker2d-medium-replay/dt.sh &
sleep 120
sh scripts/S4RL/walker2d-random/dt.sh &
wait

sh scripts/S4RL/maze2d-umaze-v1/dt.sh &
sleep 120
sh scripts/S4RL/maze2d-medium-v1/dt.sh &
sleep 120
sh scripts/S4RL/maze2d-large-v1/dt.sh &
wait
