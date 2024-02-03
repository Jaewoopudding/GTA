sh scripts/GTA_0123/maze2d-umaze-v1/bc_10.sh &
sh scripts/GTA_0123/maze2d-large-v1/bc_10.sh &
sh scripts/GTA_0123/maze2d-medium-v1/bc_10.sh &
wait

sh scripts/GTA_0123/hopper-medium/bc_10.sh &
sh scripts/GTA_0123/hopper-medium-expert/bc_10.sh &
sh scripts/GTA_0123/hopper-medium-replay/bc_10.sh &
sh scripts/GTA_0123/hopper-random/bc_10.sh &
wait

sh scripts/GTA_0123/walker2d-medium/bc_10.sh &
sh scripts/GTA_0123/walker2d-medium-expert/bc_10.sh &
sh scripts/GTA_0123/walker2d-medium-replay/bc_10.sh &
sh scripts/GTA_0123/walker2d-random/bc_10.sh &
wait

sh scripts/GTA_0123/halfcheetah-medium/bc_10.sh &
sh scripts/GTA_0123/halfcheetah-medium-expert/bc_10.sh &
sh scripts/GTA_0123/halfcheetah-medium-replay/bc_10.sh &
sh scripts/GTA_0123/halfcheetah-random/bc_10.sh &
wait