sh scripts/0117RTG/maze2d-umaze-v1/bc_100.sh &
sh scripts/0117RTG/maze2d-large-v1/bc_100.sh &
sh scripts/0117RTG/maze2d-medium-v1/bc_100.sh &
wait

sh scripts/0117RTG/hopper-medium/bc_100.sh &
sh scripts/0117RTG/hopper-medium-expert/bc_100.sh &
sh scripts/0117RTG/hopper-medium-replay/bc_100.sh &
sh scripts/0117RTG/hopper-random/bc_100.sh &
wait

sh scripts/0117RTG/walker2d-medium/bc_100.sh &
sh scripts/0117RTG/walker2d-medium-expert/bc_100.sh &
sh scripts/0117RTG/walker2d-medium-replay/bc_100.sh &
sh scripts/0117RTG/walker2d-random/bc_100.sh &
wait


sh scripts/0117RTG/halfcheetah-medium/bc_100.sh &
sh scripts/0117RTG/halfcheetah-medium-expert/bc_100.sh &
sh scripts/0117RTG/halfcheetah-medium-replay/bc_100.sh &
sh scripts/0117RTG/halfcheetah-random/bc_100.sh &
wait
