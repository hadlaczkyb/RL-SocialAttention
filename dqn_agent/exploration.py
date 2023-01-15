import numpy as np
from gym.utils import seeding
from gym import spaces


##################################
### Epsilon Greedy exploration ###
##################################
class EpsilonGreedy:
    def __init__(self, action_space, temperature=1.0, final_temperature=0.1, tau=5000):
        # Set action space
        self.action_space = action_space
        if isinstance(self.action_space, spaces.Tuple):
            self.action_space = self.action_space.spaces[0]

        # Set parameters
        self.temperature = temperature   # exploration probability at start
        self.final_temperature = min(temperature, final_temperature)    # min exploration probability
        self.tau = tau      # exponential decay rate
        self.epsilon = 0    # exploration probability
        self.time = 0       # decay steps

        # Initializations
        self.np_random = None
        self.optimal_action = None
        self.seed()

    def get_distribution(self):
        distribution = {action: self.epsilon / self.action_space.n for action in range(self.action_space.n)}
        distribution[self.optimal_action] += 1 - self.epsilon
        return distribution

    def sample(self):
        distribution = self.get_distribution()
        return self.np_random.choice(list(distribution.keys()), 1, p=np.array(list(distribution.values())))[0]

    def update(self, values):
        self.optimal_action = np.argmax(values)
        self.epsilon = self.final_temperature + (self.temperature - self.final_temperature) * \
                       np.exp(- self.time / self.tau)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_time(self, time):
        self.time = time

    def step_time(self):
        self.time = self.time + 1


##########################
### Greedy exploration ###
##########################
class Greedy:
    def __init__(self, action_space):
        # Set action space
        self.action_space = action_space
        if isinstance(self.action_space, spaces.Tuple):
            self.action_space = self.action_space.spaces[0]

        # Initializations
        self.values = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_distribution(self):
        optimal_action = np.argmax(self.values)
        return {action: 1 if action == optimal_action else 0 for action in range(self.action_space.n)}

    def update(self, values):
        self.values = values

    def sample(self):
        distribution = self.get_distribution()
        return self.np_random.choice(list(distribution.keys()), 1, p=np.array(list(distribution.values())))[0]

    def set_time(self, time):
        pass

    def step_time(self):
        pass
