import numpy as np
from gym import utils
# from gym.envs.mujoco import mujoco_env
import mujoco_py
import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

import src.envs.mujoco_env as mujoco_env


# class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
#         utils.EzPickle.__init__(self)

#     def step(self, action):
#         xposbefore = self.sim.data.qpos[0]
#         self.do_simulation(action, self.frame_skip)
#         xposafter = self.sim.data.qpos[0]
#         ob = self._get_obs()
#         reward_ctrl = -0.1 * np.square(action).sum()
#         reward_run = (xposafter - xposbefore) / self.dt
#         reward = reward_ctrl + reward_run
#         done = False
#         return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

#     def _get_obs(self):
#         return np.concatenate(
#             [
#                 self.sim.data.qpos.flat[1:],
#                 self.sim.data.qvel.flat,
#             ]
#         )

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(
#             low=-0.1, high=0.1, size=self.model.nq
#         )
#         qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
#         self.set_state(qpos, qvel)
#         return self._get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.distance = self.model.stat.extent * 0.5



class ExtendedHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)
        self.render_mode = False # 1129

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model_with_state(self, state):

        qpos = state[:self.model.nq-1]
        qvel = state[self.model.nq-1:]

        old_state = self.sim.get_state()
        xpos = old_state.qpos[0]
        qpos = np.concatenate([[xpos], qpos])

        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


