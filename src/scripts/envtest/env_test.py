import gym
import numpy as np
import torch
import argparse
from tqdm import tqdm
import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from src.data.norm import MinMaxNormalizer, normalizer_factory
from src.diffusion.utils import make_inputs,  split_diffusion_transition, construct_diffusion_model
from corl.shared.utils  import get_saved_dataset
from corl.shared.buffer import DiffusionDataset

import src.environments as environments # to register extended environments


def extended_gym_env(env_name, init_state):
    env = gym.make(env_name)
    env.reset()
    env.state = env.unwrapped.state = init_state
    return env


def instantiate_async_envs(env_name, init_states):

    vectorenv = gym.vector.AsyncVectorEnv([
        lambda : extended_gym_env(env_name, init_state) for init_state in init_states
    ])
    
    return vectorenv


class CustomgymEnv(gym.Env):
    def __init__(self, dataset, custom_start_state):
        super(CustomgymEnv, self).__init__()
        self.env = gym.make(dataset)
        self.custom_start_state = custom_start_state

    def reset(self):
        self.env.reset()
        self.env.state = self.custom_start_state  # 사용자 지정 초기 상태 설정
        return self.env.state

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def transfer_flatten_to_sim(state, env):

    sim = env.unwrapped.sim

    idx_time = 0
    idx_qpos = idx_time + 1
    idx_qvel = idx_qpos + sim.model.nq

    qpos = state[idx_qpos:idx_qpos + sim.model.nq]
    qvel = state[idx_qvel:idx_qvel + sim.model.nv]
    
    return qpos, qvel




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v2')
    args = parser.parse_args()
    

    env = args.dataset.split('-')[0]
    if env == 'hopper':
        env_nm = "HopperExt-v2"
    elif env == 'halfcheetah':
        env_nm = "HalfCheetahExt-v2"
    elif env == 'walker2d':
        env_nm = "Walker2dExt-v2"

    data = get_saved_dataset(args.dataset)
    reward_mse = []
    dynamic_mse = []
    for t in tqdm(range(data["rewards"].shape[0]-1)):
        state = data["observations"][t]
        action = data["actions"][t]
        next_state = data["next_observations"][t]
        reward = data["rewards"][t]
        
        
        try:
            timeout_flag = np.linalg.norm(
                    data["observations"][t + 1] - data["next_observations"][t]
                ) > 1e-6 \
                or data["terminals"][t] == 1.0
        except:
            timeout_flag = True
        
        if timeout_flag:
            timeout_flag = False
            continue

        env = gym.make(env_nm)
        env.reset(state = state)
        real_obs, real_reward, done, _ = env.step(action)
        dynamic_mse.append(np.square(real_obs-next_state).mean())
        reward_mse.append(np.square(real_reward-reward).mean())


    dynamic_mse = np.array(dynamic_mse)
    reward_mse = np.array(reward_mse)

    print("dynamic_mse")
    print(dynamic_mse.mean())
    print(dynamic_mse.std())
    print("reward_mse")
    print(reward_mse.mean())
    print(reward_mse.std())
        
            
            
