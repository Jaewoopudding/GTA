sh scripts/GTA_0123/maze2d-umaze-v1/iql.sh &
sh scripts/GTA_0123/maze2d-large-v1/iql.sh &
sh scripts/GTA_0123/maze2d-medium-v1/iql.sh &
wait

sh scripts/GTA_0123/walker2d-medium/iql.sh &
sh scripts/GTA_0123/walker2d-medium-expert/iql.sh &
sh scripts/GTA_0123/walker2d-medium-replay/iql.sh &
sh scripts/GTA_0123/walker2d-random/iql.sh &
wait

sh scripts/GTA_0123/halfcheetah-medium/iql.sh &
sh scripts/GTA_0123/halfcheetah-medium-expert/iql.sh &
sh scripts/GTA_0123/halfcheetah-medium-replay/iql.sh &
sh scripts/GTA_0123/halfcheetah-random/iql.sh &
wait

sh scripts/GTA_0123/hopper-medium/iql.sh &
sh scripts/GTA_0123/hopper-medium-expert/iql.sh &
sh scripts/GTA_0123/hopper-medium-replay/iql.sh &
sh scripts/GTA_0123/hopper-random/iql.sh &
wait