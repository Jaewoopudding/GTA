# sh scripts/0117reweight/maze2d-umaze-v1/iql.sh &
# sh scripts/0117reweight/maze2d-large-v1/iql.sh &
# sh scripts/0117reweight/maze2d-medium-v1/iql.sh &
# wait

sh scripts/0117reweight/walker2d-medium/iql.sh &
sh scripts/0117reweight/walker2d-medium-expert/iql.sh &
sh scripts/0117reweight/walker2d-medium-replay/iql.sh &
sh scripts/0117reweight/hopper-medium/iql.sh &
# sh scripts/0117reweight/walker2d-random/iql.sh &
wait

sh scripts/0117reweight/hopper-medium-expert/iql.sh &
sh scripts/0117reweight/hopper-medium-replay/iql.sh &
sh scripts/0117reweight/halfcheetah-medium-expert/iql.sh &
sh scripts/0117reweight/halfcheetah-medium-replay/iql.sh &
# sh scripts/0117reweight/halfcheetah-random/iql.sh &
wait

sh scripts/0117reweight/halfcheetah-medium/iql.sh &
# sh scripts/0117reweight/hopper-random/iql.sh &
wait