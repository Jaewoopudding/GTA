import gym

ENVIRONMENT_SPECS = (
    {
        'id': 'HopperExt-v2',
        'entry_point': ('src.envs.hopper:ExtendedHopperEnv'),
    },
    {
        'id': 'HalfCheetahExt-v2',
        'entry_point': ('src.envs.half_cheetah:ExtendedHalfCheetahEnv'),
    },
    {
        'id': 'Walker2dExt-v2',
        'entry_point': ('src.envs.walker2d:ExtendedWalker2dEnv'),
    },
)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ mjcd/environments/registration ] WARNING: not registering mjcd environments')
        return tuple()

# from ray.tune.registry import register_env

# from src.environments.hopper import ExtendedHalfCheetahEnv

# register_env("HopperExt-v2", lambda config: ExtendedHalfCheetahEnv())