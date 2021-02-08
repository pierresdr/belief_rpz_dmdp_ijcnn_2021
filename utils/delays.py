import numpy as np
import random as rnd
from gym import Wrapper, spaces
import copy


class JumpProcess:
    def __init__(self, init, max_value=50):
        self.current = init
        self.init = init
        self.max = max_value
        
    def sample(self):
        raise NotImplementedError

    def reset(self):
        self.current = self.init


class ConstantDelay(JumpProcess):
    def __init__(self, init):
        super().__init__(init, max_value=init)

    def sample(self):
        return self.current, 1

# The minimum delay is 0 for the following process
# class NormedPositiveCompoundBernoulliProcess(JumpProcess):

#     def __init__(self, p, init=0, max_value=50):
#         """
#         A compound Bernoulli process which is forced positive by normalizing
#         probabilities. The process is built as follows:
#             Y_t = sum_{i=0}^t (Z_i)
#         where (Z_i) are such that:
#             p( Z_t=a | Y_t=b ) = 1/C . (1-p)p^(1-a), for a \in [-b,1]
#                                 it is 0 otherwise.

#         Arguments:
#             p (float): the probability of a downard jump of size one
#                     before normalization
#             init (int): the initial value of the series
#         """
#         super().__init__(init, max_value)
#         self.p = p
#         self.series = [init]
        
#     def sample(self):
#         C = 1 - self.p**(self.current+2)
#         u = rnd.random()
#         jump = 1-int(np.log(1-C*u)/np.log(self.p))
#         self.current = self.current+jump

#         if self.current>self.max:
#             self.current = self.max
#             n_obs = 1
#         else:
#             n_obs = 1-jump
#         self.series.append(self.current)
#         return self.current, n_obs

# The minimum delay is one for the following process


class NormedPositiveCompoundBernoulliProcess(JumpProcess):

    def __init__(self, p, init=1, max_value=50):
        """
        A compound Bernoulli process which is forced positive by normalizing
        probabilities. The process is built as follows:
            Y_t = sum_{i=0}^t (Z_i)
        where (Z_i) are such that:
            p( Z_t=a | Y_t=b ) = 1/C . (1-p)p^(1-a), for a \in [-b,1]
                                it is 0 otherwise.

        Arguments:
            p (float): the probability of a downard jump of size one
                    before normalization
            init (int): the initial value of the series
        """
        super().__init__(init, max_value)
        self.p = p
        self.series = [init]
        
    def sample(self):
        C = 1 - self.p**(self.current+1)
        u = rnd.random()
        jump = 1-int(np.log(1-C*u)/np.log(self.p))
        self.current = self.current+jump

        if self.current > self.max:
            self.current = self.max
            n_obs = 1
        else:
            n_obs = 1-jump
        self.series.append(self.current)
        return self.current, n_obs
        

class PositiveCompoundBernoulliProcess(JumpProcess):

    def __init__(self, p, init=1, max_value=50):
        """
        A compound Bernoulli process which is forced positive by truncation of the 
        negative probabilities. The process is built as follows:
            Y_t = sum_{i=0}^t (Z_i)
        where (Z_i) are such that:
            p( Z_t=a | Y_t=b ) = (1-p)p^(1-a), for a \in [-b+1,1]
                                (1-p)p^(b+1) + p^(b+2) for a=-b

        Arguments:
            p (float): the probability of a downard jump of size one
                    before normalization
            init (int): the initial value of the series
        """
        super().__init__(init, max_value)
        self.p = p
        self.series = [init]

    def sample(self):
        u = rnd.random()
        jump = 1-int(np.log(1-u)/np.log(self.p))
        jump = max(jump, -self.current+1)
        self.current = min(self.current+jump, self.max)

        if self.current > self.max - 1:
            self.current = self.max - 1
            n_obs = 1
        else:
            n_obs = 1-jump
        self.series.append(self.current)
        return self.current, n_obs


class DelayWrapper(Wrapper):
    def __init__(self, env, delay=0, stochastic_delays=False, p_delay=0.70, max_delay=50):
        super(DelayWrapper, self).__init__(env)

        # Delay Process initialization
        self.stochastic_delays = stochastic_delays
        if stochastic_delays:
            self.delay = NormedPositiveCompoundBernoulliProcess(p_delay, delay, max_value=max_delay)
        else: 
            self.delay = ConstantDelay(delay)

        # Create State and Observation Space
        self.state_space = self.observation_space

        if isinstance(self.action_space, spaces.Discrete):
            size = self.action_space.n*self.delay.max
            stored_actions = spaces.Discrete(size)
        else:
            high = np.tile(self.action_space.high, self.delay.max) 
            low = np.tile(self.action_space.high, self.delay.max) 
            shape = [self.delay.max*i for i in self.action_space.shape]
            dtype = self.action_space.dtype
            stored_actions = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

        self.observation_space = spaces.Dict({
                'last_obs': copy.deepcopy(self.observation_space),
                'stored_actions': stored_actions,
        })

        # Delay Variables initialization
        self._hidden_obs = None
        self._reward_stock = None
        self.extended_obs = None

    def reset(self, **kwargs):
        # Reset the underlying Environment
        obs = self.env.reset(**kwargs)

        # Reset the Delay Process
        self.delay.reset()

        # Prepare the first Extended State by acting randomly until it is complete:
        # 1. Reset all Variables
        self._hidden_obs = [0 for _ in range(self.delay.current)]
        self.extended_obs = [0 for _ in range(self.delay.current+1)]
        self._reward_stock = np.array([0 for _ in range(self.delay.current)])
        # 2. The state in the first Extended State is the reset state of the Environment
        self._hidden_obs.append(obs)
        self.extended_obs[0] = self._hidden_obs[-1]
        # 3. Sample and Execute action to fill the rest of the Extended State
        obs_before_start = self.delay.current
        while obs_before_start > 0:
            # 3a. Sample and Execute current action
            action = self.action_space.sample()
            _, _, done, info = self.step(action)
            # 3b. If the Environment happen to fail before the first Extended State is built, notify it
            if done:
                logger.warn("The environment failed before delay timesteps.")
                return self.reset()

            obs_before_start -= info[0]

        # Return (State, Actions)
        return self.extended_obs[0], np.array(self.extended_obs[1:])

    def step(self, action):
        # Execute the Action in the underlying environment
        obs, reward, done, info = self.env.step(action)

        # Sample new delay
        _, n_obs = self.delay.sample()

        # Get current state
        self._hidden_obs.append(obs)

        # Update extended state and hidden variables
        self.extended_obs.append(action)
        hidden_output = None
        if n_obs > 0:
            self.extended_obs[0] = self._hidden_obs[n_obs]
            del self.extended_obs[1:(1+n_obs)]
            hidden_output = np.array(self._hidden_obs[1:(1+n_obs)])
            del self._hidden_obs[:n_obs]

        # Update the reward array and determine current reward output
        self._reward_stock = np.append(self._reward_stock, reward)
        if done:
            reward_output = self._reward_stock
            # reward_output = self._reward_stock # -> in this case, the sum is to be done in the algorithm
        else:
            reward_output = self._reward_stock[:n_obs]
        self._reward_stock = np.delete(self._reward_stock, range(n_obs))

        # Shaping the output
        output = (self.extended_obs[0], np.array(self.extended_obs[1:]))

        return output, reward_output, done, (n_obs, hidden_output)







