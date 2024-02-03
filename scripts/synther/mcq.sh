# sh scripts/synther/walker2d-medium/mcq.sh &
# sh scripts/synther/halfcheetah-medium-expert/mcq.sh &
# sh scripts/synther/halfcheetah-medium-replay/mcq.sh &
# sh scripts/synther/halfcheetah-random/mcq.sh &
# sh scripts/synther/hopper-medium/mcq.sh &
# sh scripts/synther/hopper-medium-expert/mcq.sh &
# sh scripts/synther/hopper-medium-replay/mcq.sh &
# sh scripts/synther/hopper-random/mcq.sh &
# wait

sh scripts/synther/halfcheetah-medium/mcq.sh &
sh scripts/synther/walker2d-medium-expert/mcq.sh &
sh scripts/synther/walker2d-medium-replay/mcq.sh &
sh scripts/synther/walker2d-random/mcq.sh &

# sh scripts/synther/maze2d-umaze-v1/mcq.sh &
# sh scripts/synther/maze2d-large-v1/mcq.sh &
# sh scripts/synther/maze2d-medium-v1/mcq.sh &
wait