import numpy as np
from gym import utils
import mujoco_py
import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

import src.envs.mujoco_env as mujoco_env



# class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     def __init__(self):
#         mujoco_env.MujocoEnv.__init__(self, "hopper.xml", 4)
#         utils.EzPickle.__init__(self)

#     def step(self, a):
#         posbefore = self.sim.data.qpos[0]
#         self.do_simulation(a, self.frame_skip)
#         posafter, height, ang = self.sim.data.qpos[0:3]
#         alive_bonus = 1.0
#         reward = (posafter - posbefore) / self.dt
#         reward += alive_bonus
#         reward -= 1e-3 * np.square(a).sum()
#         s = self.state_vector()
#         done = not (
#             np.isfinite(s).all()
#             and (np.abs(s[2:]) < 100).all()
#             and (height > 0.7)
#             and (abs(ang) < 0.2)
#         )
#         ob = self._get_obs()
#         return ob, reward, done, {}

#     def _get_obs(self):
#         return np.concatenate(
#             [self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)]
#         )

#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(
#             low=-0.005, high=0.005, size=self.model.nq
#         )
#         qvel = self.init_qvel + self.np_random.uniform(
#             low=-0.005, high=0.005, size=self.model.nv
#         )
#         self.set_state(qpos, qvel)
#         return self._get_obs()

#     def viewer_setup(self):
#         self.viewer.cam.trackbodyid = 2
#         self.viewer.cam.distance = self.model.stat.extent * 0.75
#         self.viewer.cam.lookat[2] = 1.15
#         self.viewer.cam.elevation = -20



class ExtendedHopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "hopper.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate(
            [self.sim.data.qpos.flat[1:], np.clip(self.sim.data.qvel.flat, -10, 10)]
        )

    def reset_model_with_state(self, state):

        qpos = state[:self.model.nq-1]
        qvel = state[self.model.nq-1:]

        qvel = np.clip(qvel, -10, 10)

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
            low=-0.005, high=0.005, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
