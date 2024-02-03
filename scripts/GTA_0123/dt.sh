sh scripts/GTA_0123/hopper-medium/dt.sh &
sleep 120
sh scripts/GTA_0123/hopper-medium-expert/dt.sh &
sleep 120
sh scripts/GTA_0123/hopper-medium-replay/dt.sh &
sleep 120
sh scripts/GTA_0123/hopper-random/dt.sh &
wait

sh scripts/GTA_0123/walker2d-medium/dt.sh &
sleep 120
sh scripts/GTA_0123/walker2d-medium-expert/dt.sh &
sleep 120
sh scripts/GTA_0123/walker2d-medium-replay/dt.sh &
sleep 120
sh scripts/GTA_0123/walker2d-random/dt.sh &
wait

sh scripts/GTA_0123/halfcheetah-medium/dt.sh &
sleep 120
sh scripts/GTA_0123/halfcheetah-medium-expert/dt.sh &
sleep 120
sh scripts/GTA_0123/halfcheetah-medium-replay/dt.sh &
sleep 120
sh scripts/GTA_0123/halfcheetah-random/dt.sh &
wait 

sh scripts/GTA_0123/maze2d-umaze-v1/dt.sh &
sleep 120
sh scripts/GTA_0123/maze2d-large-v1/dt.sh &
sleep 120
sh scripts/GTA_0123/maze2d-medium-v1/dt.sh &
wait
