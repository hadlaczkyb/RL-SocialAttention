import torch
import torch.nn as nn
import numpy as np

from gym import spaces
from utils.config import Configurable
from utils.select_device import device as dev
from ppo_agent.actor_critic import ActorCritic
from ppo_agent.rollout_buffer import RolloutBuffer


# noinspection PySingleQuotedDocstring
#######################
### PPO agent class ###
#######################
class PPOAgent(Configurable):
    def __init__(self, env, lr_actor=3e-4, lr_critic=1e-3, weight_decay=1e-5, agent_config=None, policy_config=None):
        super().__init__(agent_config)

        # Set agent and nn. model configurations
        self.config = agent_config
        self.policy_config = policy_config
        # Set environment
        self.env = env
        # Create 'Rollout Buffer'
        self.buffer = RolloutBuffer()

        # Set agent parameters
        self.eps_clip = self.config["eps_clip"]   # clip ratio: how far can the new policy go from the old policy while
        # still profiting
        self.gamma = self.config["gamma"]         # discount factor [0; 1]
        self.lam = self.config["lam"]             # smoothing factor [0; 1]
        self.K_epochs = self.config["K_epochs"]   # number of epochs for policy update
        self.action_std_init = self.config["action_std_init"]  # initial standard deviation of random action selection
        self.action_std_decay_rate = self.config["action_std_decay_rate"]   # decay rate of action selection std.
        self.min_action_std = self.config["min_action_std"]    # minimal standard deviation of random action selection
        self.action_std_decay_freq = int(1e3)     # decay frequency in timesteps of action selection std.
        self.update_timestep = self.config["update_timestep"]   # policy update timesteps

        # Set if environment has Discrete action space or Continuous (Box)
        if isinstance(env.action_space, spaces.Discrete):
            self.has_continuous_action_space = False
        else:
            self.has_continuous_action_space = True

        # If the environment has Continuous action space, initialize standard deviation of random action selection
        if self.has_continuous_action_space:
            self.action_std = self.action_std_init

        # Set device
        self.device = dev

        # Create policy: Actor and Critic networks
        self.policy = ActorCritic(
            env,
            self.has_continuous_action_space,
            self.action_std_init,
            self.policy_config
        ).to(self.device)

        # Set MSE loss function
        self.mse_loss = nn.MSELoss()

        # Set ADAM optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor, 'weight_decay': weight_decay},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic, 'weight_decay': weight_decay}
        ])

        # Set old policy
        self.policy_old = ActorCritic(
            env,
            self.has_continuous_action_space,
            self.action_std_init,
            self.policy_config
        ).to(self.device)
        self.policy_old.actor.load_state_dict(self.policy.actor.state_dict())
        self.policy_old.critic.load_state_dict(self.policy.critic.state_dict())

        # Initializations
        self.training = True
        self.previous_state = None
        self.directory = None
        self.steps = 0

    @classmethod
    def default_config(cls):
        return dict(eps_clip=0.2,
                    gamma=0.99,
                    lam=0.95,
                    K_epochs=100,
                    action_std_init=0.6,
                    action_std_decay_rate=0.05,
                    min_action_std=0.1,
                    update_timestep=256
                    )

    #*****************************************************************
    # Put data into Rollout Buffer, update networks, decay action std.
    #*****************************************************************
    def record(self, state=None, action=None, reward=None, next_state=None, done=None, info=None):
        if not self.training:
            return

        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
            value = self.policy.critic(state)

        # Put value, reward and done into the Rollout Buffer
        self.buffer.values.append(value)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

        self.steps = self.steps + 1
        # Update networks
        if self.steps % self.update_timestep == 0:
            with torch.no_grad():
                next_state = torch.tensor(np.array([next_state]), dtype=torch.float).to(self.device)
                value = self.policy.critic(next_state)
            self.buffer.values.append(value)
            self.update()

        # Decay action std.
        if self.policy.has_continuous_action_space and self.steps % self.action_std_decay_freq == 0:
            self.decay_action_std(self.action_std_decay_rate, self.min_action_std)

    #***************************************************
    # Set action std. in case of continuous action space
    #***************************************************
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            pass

    #*****************************************************
    # Decay action std. in case of continuous action space
    #*****************************************************
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
            self.set_action_std(self.action_std)
        else:
            pass

    #*********************************
    # Select action in the given state
    #*********************************
    def select_action(self, state):
        self.previous_state = state
        if self.has_continuous_action_space:
            with torch.no_grad():
                # Reshape state to [batch; N vehicles; features]
                # state = torch.FloatTensor(np.array([state])).to(self.device)
                state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            # Put current state, action and logarithmic action probabilities into the Rollout Buffer
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():   # TODO: - why is this torch.no_grad() necessary here?
                # Reshape state to [batch; N vehicles; features]
                # state = torch.FloatTensor(np.array([state])).to(self.device)
                state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            # Put current state, action and logarithmic action probabilities into the Rollout Buffer
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        '''
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        '''

        # Generalized Advantage Estimation of returns
        rewards = []
        gae = 0
        for i in reversed(range(len(self.buffer.rewards))):
            mask = not self.buffer.is_terminals[i] 
            delta = self.buffer.rewards[i] + self.gamma * self.buffer.values[i+1] * mask - self.buffer.values[i]
            gae = delta + self.gamma * self.lam * mask * gae
            rewards.insert(0, gae + self.buffer.values[i])

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Match state values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            advantages = (advantages - advantages.mean())/(advantages.std() + 1e-10)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO = actor's loss + critic's loss + entropy loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.001 * dist_entropy
            # - 0.01 * torch.mean(-torch.exp(logprobs) * logprobs)

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            for param in self.policy.actor.parameters():
                param.grad.data.clamp_(-1, 1)
            for param in self.policy.critic.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.actor.load_state_dict(self.policy.actor.state_dict())
        self.policy_old.critic.load_state_dict(self.policy.critic.state_dict())

        # Clear Rollout Buffer
        self.buffer.clear()

    # def get_batch_state_values(self, states):
    #     values, actions = self.policy.actor(torch.tensor(states, dtype=torch.float).to(self.device)).max(1)
    #     return values.data.cpu().numpy(), actions.data.cpu().numpy()

    # ***********************************************************
    # Get batched state-action values (necessary for PPOGraphics)
    # ***********************************************************
    def get_batch_state_action_values(self, states):
        # values, actions = self.policy.actor(torch.tensor(np.array(states), dtype=torch.float).to(self.device)).max(1)
        # return values.data.cpu().numpy(), actions.data.cpu().numpy()
        return self.policy.actor(torch.tensor(states, dtype=torch.float).to(self.device)).data.cpu().numpy()

    #****************************************************
    # Get state-action values (necessary for PPOGraphics)
    #****************************************************
    def get_state_action_values(self, state):
        return self.get_batch_state_action_values([state])[0]

    #***************************************************************************
    # Return selected action in the given state (necessary for TrainAndEvaluate)
    #***************************************************************************
    def plan(self, state):
        return [self.select_action(state)]

    def seed(self, seed=None):
        pass

    def reset(self):
        pass

    def eval(self):
        self.training = False

    def set_directory(self, directory):
        self.directory = directory

    def save(self, filename):
        state = {'state_dict_actor': self.policy_old.actor.state_dict(),
                 'state_dict_critic': self.policy_old.critic.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)

        self.policy_old.actor.load_state_dict(checkpoint['state_dict_actor'])
        self.policy_old.critic.load_state_dict(checkpoint['state_dict_critic'])

        self.policy.actor.load_state_dict(checkpoint['state_dict_actor'])
        self.policy.critic.load_state_dict(checkpoint['state_dict_critic'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename
