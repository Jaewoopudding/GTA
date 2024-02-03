from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Uniform, Beta, Bernoulli


import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".aug-project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)


class S4RLAugmentation:
    def __init__(self, 
                 type, 
                 std_scale=1e-2, 
                 uniform_scale=1e-2, 
                 mix_up_scale=0.4,
                 adv_scale=1e-2, 
                 ):
        '''
        Trajectory와 Transition 어떤 경우에서도  사용할 수 있는 Augmentation Function.
        Gaussian Noise와 Amplitude Scaling은 RAD의 그것과 동일하다.
        adversarial attack은 trajectory optimizaiton model에서는 작동하지 않는다. 
        '''
        self.normal_distribution = Normal(loc=0, scale=std_scale)
        self.uniform_distribution = Uniform(low=-uniform_scale, high=uniform_scale)
        self.beta_distribution = Beta(mix_up_scale, mix_up_scale)
        self.adv_sacle = adv_scale
        self.type = type
        self.trajectory = False
        if type == 'gaussian_noise':
            self.aug_fn = partial(self.gaussian_noise)
        elif type == 'uniform_noise':
            self.aug_fn = partial(self.uniform_noise)
        elif type == 'amplitude_scaling':
            self.aug_fn = partial(self.amplitude_scaling)
        elif type == 'amplitude_scaling_m':
            self.aug_fn = partial(self.amplitude_scaling_m)
        elif type == 'dimension_dropout':
            self.aug_fn = partial(self.dimension_dropout)
        elif type == 'state_mix_up':
            self.aug_fn = partial(self.state_mix_up)
        elif type == 'adversarial_state_training':
            self.aug_fn = partial(self.adversarial_state_training)
        elif type == 'identical' or type is None:
            self.aug_fn = partial(self.identical)
        else:
            raise NotImplementedError(f'{type} <- 그런 이름을 가진 Augmentation은 없답니다.')
        print("S4RL Augmentation 활성화 완료")
        print(f"Augmentation Type: {type}")
        # dim_drop이랑 state_switch는 physical realizability를 심각하게 손상시키는 두개의 방법
        # state switch는 안 할거임. -> extensive 한 hyperparameter serach가 필요함. 
        # dim_drop은 아직 하고 있음. 구현이 생각보다 까다로움
    
    def __call__(self, state, action=None, next_state=None, q_function=None):
        return self.aug_fn(state, action, next_state, q_function)

    def trajectory_flag(self): # 입력이 (s a r s) transition이 아니라, (s, s, s .., s) 로 변함
        self.trajectory = True
    
    def identical(self, state, action=None, next_state=None, q_function=None):
        '''
        
        '''
        if self.trajectory:
            return state
        return state, next_state

    def gaussian_noise(self, state, action=None, next_state=None, q_function=None):
        '''
        
        '''
        if isinstance(state, np.ndarray):
            state += self.normal_distribution.sample(state.shape).detach().numpy()
            if self.trajectory:
                return state
            next_state += self.normal_distribution.sample(state.shape).detach().numpy()
            return state, next_state

        state += self.normal_distribution.sample(state.shape).to(state.device)
        if self.trajectory:
            return state
        next_state += self.normal_distribution.sample(state.shape).to(state.device)
        return state, next_state

    def uniform_noise(self, state, action=None, next_state=None, q_function=None):
        '''

        '''
        if isinstance(state, np.ndarray):
            state += self.uniform_distribution.sample(state.shape).detach().numpy()
            if self.trajectory:
                return state
            next_state += self.uniform_distribution.sample(state.shape).detach().numpy()
            return state, next_state

        state += self.uniform_distribution.sample(state.shape).to(state.device)
        if self.trajectory:
            return state
        next_state += self.uniform_distribution.sample(state.shape).to(state.device)
        return state, next_state


    def amplitude_scaling(self, state, action=None, next_state=None, q_function=None):
        '''
        
        '''
        if isinstance(state, np.ndarray):
            state *= (1 + self.uniform_distribution.sample(self._dim_matcher(state)).detach().numpy())
            if self.trajectory:
                return state
            next_state *= (1 + self.uniform_distribution.sample(self._dim_matcher(next_state)).detach().numpy())
            return state, next_state

        state *= (1 + self.uniform_distribution.sample(self._dim_matcher(state)).to(state.device))
        if self.trajectory:
            return state
        next_state *= (1 + self.uniform_distribution.sample(self._dim_matcher(next_state)).to(state.device))
        return state, next_state

    def amplitude_scaling_m(self, state, action=None, next_state=None, q_function=None):
        '''
        
        '''
        if isinstance(state, np.ndarray):
            state *= (1 + self.uniform_distribution.sample(state.shape).detach().numpy())
            if self.trajectory:
                return state
            next_state *= (1 + self.uniform_distribution.sample(state.shape).detach().numpy())
            return state, next_state
    
        state *= (1 + self.uniform_distribution.sample(state.shape).to(state.device))
        if self.trajectory:
            return state
        next_state *= (1 + self.uniform_distribution.sample(state.shape).to(state.device))
        return state, next_state
    
    def dimension_dropout(self, state, action=None, next_state=None, q_function=None):
        raise NotImplementedError("아직 안함")

    def state_mix_up(self, state, action=None, next_state=None, q_function=None):
        assert not self.trajectory
        if isinstance(state, np.ndarray):
            raise NotImplementedError
        coef = self.beta_distribution.sample(self._dim_matcher(state)).to(state.device)
        return (state * coef) + (next_state * (1 - coef)), next_state


    def adversarial_state_training(self, state, action, next_state=None, q_function=None):
        '''
        Q function의 방향으로 adversarial attack을 수행하는 augmentation
        Trajectory Level에서는 사용할 수 없다. 
        '''
        assert not self.trajectory
        if isinstance(state, np.ndarray):
            raise NotImplementedError
        state.requires_grad_()
        # state_action = torch.cat([state, action], dim=-1)
        q_value = q_function(state, action).mean()
        q_value.backward()
        q_function.zero_grad()
        grad = state.grad
        state = state.clone() + self.adv_sacle * grad

        return state, next_state
    
    
    def _dim_matcher(self, state):
        if isinstance(state, np.ndarray):
            dim = state.ndim
        elif isinstance(state, torch.Tensor):
            dim = state.dim()
        else: 
            raise ValueError()

        shape = state.shape
        if dim == 2: # for one_step transition
            return [shape[0], 1]
        elif dim == 3:
            return [shape[0], 1, 1]
        raise ValueError(f"Input is not a transition, nor trajectory. it has {dim} dimension")