sh scripts/0117RTG/maze2d-umaze-v1/iql.sh &
sh scripts/0117RTG/maze2d-large-v1/iql.sh &
sh scripts/0117RTG/maze2d-medium-v1/iql.sh &
wait

sh scripts/0117RTG/walker2d-medium/iql.sh &
sh scripts/0117RTG/walker2d-medium-expert/iql.sh &
sh scripts/0117RTG/walker2d-medium-replay/iql.sh &
sh scripts/0117RTG/walker2d-random/iql.sh &
wait

sh scripts/0117RTG/halfcheetah-medium/iql.sh &
sh scripts/0117RTG/halfcheetah-medium-expert/iql.sh &
sh scripts/0117RTG/halfcheetah-medium-replay/iql.sh &
sh scripts/0117RTG/halfcheetah-random/iql.sh &
wait

sh scripts/0117RTG/hopper-medium/iql.sh &
sh scripts/0117RTG/hopper-medium-expert/iql.sh &
sh scripts/0117RTG/hopper-medium-replay/iql.sh &
sh scripts/0117RTG/hopper-random/iql.sh &
wait