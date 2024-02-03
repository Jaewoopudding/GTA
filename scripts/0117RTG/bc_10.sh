sh scripts/0117RTG/maze2d-umaze-v1/bc_10.sh &
sh scripts/0117RTG/maze2d-large-v1/bc_10.sh &
sh scripts/0117RTG/maze2d-medium-v1/bc_10.sh &
wait

sh scripts/0117RTG/hopper-medium/bc_10.sh &
sh scripts/0117RTG/hopper-medium-expert/bc_10.sh &
sh scripts/0117RTG/hopper-medium-replay/bc_10.sh &
sh scripts/0117RTG/hopper-random/bc_10.sh &
wait

sh scripts/0117RTG/walker2d-medium/bc_10.sh &
sh scripts/0117RTG/walker2d-medium-expert/bc_10.sh &
sh scripts/0117RTG/walker2d-medium-replay/bc_10.sh &
sh scripts/0117RTG/walker2d-random/bc_10.sh &
wait

sh scripts/0117RTG/halfcheetah-medium/bc_10.sh &
sh scripts/0117RTG/halfcheetah-medium-expert/bc_10.sh &
sh scripts/0117RTG/halfcheetah-medium-replay/bc_10.sh &
sh scripts/0117RTG/halfcheetah-random/bc_10.sh &
wait