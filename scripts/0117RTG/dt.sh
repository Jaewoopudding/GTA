sh scripts/0117RTG/hopper-medium/dt.sh &
sleep 100
sh scripts/0117RTG/hopper-medium-expert/dt.sh &
sleep 100
sh scripts/0117RTG/hopper-medium-replay/dt.sh &
sleep 100
sh scripts/0117RTG/hopper-random/dt.sh &
wait

sh scripts/0117RTG/walker2d-medium/dt.sh &
sleep 100
sh scripts/0117RTG/walker2d-medium-expert/dt.sh &
sleep 100
sh scripts/0117RTG/walker2d-medium-replay/dt.sh &
sleep 100
sh scripts/0117RTG/walker2d-random/dt.sh &
wait

sh scripts/0117RTG/halfcheetah-medium/dt.sh &
sleep 100
sh scripts/0117RTG/halfcheetah-medium-expert/dt.sh &
sleep 100
sh scripts/0117RTG/halfcheetah-medium-replay/dt.sh &
sleep 100
sh scripts/0117RTG/halfcheetah-random/dt.sh &
wait 

sh scripts/0117RTG/maze2d-umaze-v1/dt.sh &
sleep 100
sh scripts/0117RTG/maze2d-large-v1/dt.sh &
sleep 100
sh scripts/0117RTG/maze2d-medium-v1/dt.sh &
wait
