import torch
import numpy as np

from torch.nn import functional as F
from utils.config import Configurable
from utils.select_device import device as dev
from models.models import create_model, size_model_config
from dqn_agent.exploration import EpsilonGreedy, Greedy
from dqn_agent.memory import Transition, ReplayMemory


#######################
### DQN agent class ###
#######################
class DQNAgent(Configurable):
    def __init__(self, env, lr=1e-3, weight_decay=1e-5, agent_config=None, model_config=None):
        super().__init__(agent_config)

        # Set agent and nn. model configurations
        self.config = agent_config
        self.model_config = model_config
        # Set environment
        self.env = env

        # Set agent parameters
        self.gamma = self.config["gamma"]         # Replay Memory discount factor [0; 1]
        self.n_steps = self.config["n_steps"]     # Replay Memory steps
        self.memory_capacity = self.config["memory_capacity"]     # Replay Memory capacity

        self.exploration = agent_config["exploration"]

        self.target_update = self.config["target_update"]  # number of steps between two Target network update
        self.batch_size = self.config["batch_size"]        # batch size
        # TODO: is this 'double' necessary?
        if self.config["double"]:
            self.double = self.config["double"]
        else:
            self.double = False

        # Create Replay Memory
        self.memory = ReplayMemory(self.gamma, self.n_steps, self.memory_capacity)
        # Set exploration policy
        if self.exploration["method"] == "EpsilonGreedy":
            self.exploration_policy = EpsilonGreedy(
                self.env.action_space,
                self.exploration["temperature"],
                self.exploration["final_temperature"],
                self.exploration["tau"]
            )
        else:
            raise ValueError("Invalid exploration method!")

        # Initializations
        self.training = True
        self.previous_state = None
        self.directory = None
        self.steps = 0

        # Create Value and Target networks
        size_model_config(self.env, self.model_config)
        self.value_net = create_model(self.model_config)
        self.target_net = create_model(self.model_config)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()      # Set target network to evaluation mode
        # TODO: logger.debug("Number of trainable parameters: {}".format(trainable_parameters(self.value_net)))

        # Set device
        self.device = dev
        # Copy the networks to the selected device
        self.value_net.to(self.device)
        self.target_net.to(self.device)

        # Set MSE loss function
        self.loss_function = F.mse_loss
        # self.loss_function = nn.MSELoss()

        # Set ADAM optimizer
        self.optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    @classmethod
    def default_config(cls):
        return dict(batch_size=100,
                    gamma=0.99,
                    n_steps=1,
                    memory_capacity=50000,
                    exploration=dict(method="EpsilonGreedy",
                                     temperature=1.0,
                                     final_temperature=0.05,
                                     tau=15000),
                    target_update=1,
                    double=True)

    #*********************************************
    # Put data into Replay Memory, update networks
    #*********************************************
    def record(self, state, action, reward, next_state, done, info):
        if not self.training:
            return

        # Put state, action, reward, next state, done and info into the Replay Memory
        if isinstance(state, tuple) and isinstance(action, tuple):
            [self.memory.push(agent_state, agent_action, reward, agent_next_state, done, info)
             for agent_state, agent_action, agent_next_state in zip(state, action, next_state)]
        else:
            self.memory.push(state, action, reward, next_state, done, info)

        batch = self.sample_minibatch()
        if batch:
            loss, _, _ = self.compute_bellman_residual(batch)
            # Update networks
            self.step_optimizer(loss)
            self.update_target_network()

    #*******************************
    # Select action in a given state
    #*******************************
    def act(self, state, step_exploration_time=True):
        self.previous_state = state

        # TODO: - AttributeError: 'Greedy' object has no attribute 'step_time'
        if step_exploration_time:
            self.exploration_policy.step_time()

        # Handle multi-agent observations
        if isinstance(state, tuple):
            return tuple(self.act(agent_state, step_exploration_time=False) for agent_state in state)

        # Handle single-agent observations
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)

        return self.exploration_policy.sample()

    #***************************************************************************
    # Return selected action in the given state (necessary for TrainAndEvaluate)
    #***************************************************************************
    def plan(self, state):
        return [self.act(state)]

    def sample_minibatch(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        return Transition(*zip(*transitions))

    def update_target_network(self):
        self.steps = self.steps + 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute and concatenate batch elements
        if not isinstance(batch.state, torch.Tensor):
            state = torch.cat(tuple(torch.tensor(np.array([batch.state]), dtype=torch.float))).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor(np.array([batch.next_state]), dtype=torch.float))).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.value_net(batch.state)
        state_action_values = state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.double:
                    # Double Q-learning: pick best actions from policy network
                    _, best_actions = self.value_net(batch.next_state).max(1)
                    # Double Q-learning: estimate action values from target network
                    best_values = self.target_net(batch.next_state).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    best_values, _ = self.target_net(batch.next_state).max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward + self.gamma * next_state_values

        # Compute loss
        loss = self.loss_function(state_action_values, target_state_action_value)
        return loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        values, actions = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        # values, actions = self.value_net(torch.tensor(np.array(states), dtype=torch.float).to(self.device)).max(1)
        # return values.data.cpu().numpy(), actions.data.cpu().numpy()
        return self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()

    def step_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def seed(self, seed=None):
        return self.exploration_policy.seed(seed)

    def reset(self):
        pass

    def set_directory(self, directory):
        self.directory = directory

    def action_distribution(self, state):
        self.previous_state = state
        values = self.get_state_action_values(state)
        self.exploration_policy.update(values)
        return self.exploration_policy.get_distribution()

    def save(self, filename):
        state = {'state_dict': self.value_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_time(self, time):
        self.exploration_policy.set_time(time)

    #*************************************
    # Evaluate agent
    #   - always select the optimal action
    #*************************************
    def eval(self):
        self.training = False
        # For selecting the optimal action use Greed exploration policy
        self.exploration["method"] = "Greedy"
        self.exploration_policy = Greedy(
                self.env.action_space
            )

    def get_state_action_values(self, state):
        return self.get_batch_state_action_values([state])[0]

    def get_state_values(self, state):
        values, actions = self.get_batch_state_values([state])
        return values[0], actions[0]
