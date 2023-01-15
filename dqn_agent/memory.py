import random
from collections.__init__ import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'info'))


class ReplayMemory:
    def __init__(self, gamma=0.99, n_steps=1, memory_capacity=1e5, transition_type=Transition):
        # Set parameters
        self.gamma = gamma          # discount factor [0; 1]
        self.n_steps = n_steps      # number of steps to take
        self.capacity = int(memory_capacity)    # replay memory capacity
        self.transition_type = transition_type
        # Initializations
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.position = len(self.memory) - 1
        elif len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]
        self.memory[self.position] = self.transition_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, collapsed=True):
        if self.n_steps == 1:
            # Directly sample transition
            return random.sample(self.memory, batch_size)
        else:
            # Sample initial transition indices
            indices = random.sample(range(len(self.memory)), batch_size)
            # Get the batch of n consecutive transitions starting from sampled indices
            all_transitions = [self.memory[i:i+self.n_steps] for i in indices]
            # Collapse transitions
            return map(self.collapse_n_steps, all_transitions) if collapsed else all_transitions

    def collapse_n_steps(self, transitions):
        state, action, cumulated_reward, next_state, done, info = transitions[0]
        discount = 1
        for transition in transitions[1:]:
            if done:
                break
            else:
                _, _, reward, next_state, done, info = transition
                discount = self.gamma * discount
                cumulated_reward = discount * reward + cumulated_reward
        return state, action, cumulated_reward, next_state, done, info

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.capacity

    def is_empty(self):
        return len(self.memory) == 0
