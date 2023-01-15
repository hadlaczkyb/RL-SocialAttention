import torch
import torch.nn as nn
import numpy as np

from models.models import create_model, size_model_config
from torch.distributions import MultivariateNormal, Categorical
from utils.select_device import device as dev


##################################
### Actor-critic class for PPO ###
##################################
class ActorCritic(nn.Module):
    def __init__(self, env, has_continuous_action_space, action_std_init, policy_config=None):
        super().__init__()

        # Set device
        self.device = dev
        # Set if the environment has Discrete action space or Continuous (Box)
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = env.action_space.shape[0]
            self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(self.device)

        # Create Actor and Critic networks
        if policy_config is None:
            self.actor = create_model({"type": "FullyConnectedNetwork"})
            self.critic = create_model({"type": "FullyConnectedNetwork"})
        else:
            size_model_config(env, policy_config["actor_network"])
            size_model_config(env, policy_config["critic_network"])
            self.actor = create_model(policy_config["actor_network"])
            self.critic = create_model(policy_config["critic_network"])

        # In case of Discrete action space apply Softmax to the last dimension (because of the Categorical distribution)
        '''
        if has_continuous_action_space is False:
            self.actor = nn.Sequential(
                self.actor,
                nn.Softmax(dim=-1)
            )
        '''

    #***************************************************
    # Set action std. in case of continuous action space
    #***************************************************
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            pass

    def forward(self):
        raise NotImplementedError

    #*****************************************
    # Get action from actor in the given state
    #*****************************************
    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # For continuous action spaces, use Multivariate Normal dist. for actions
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            # For discrete action spaces, use Categorical dist. for actions
            dist = Categorical(action_probs)

        # Sample action from the given distribution
        action = dist.sample()
        # Get the logarithmic probability of the selected action
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    #**********************************************
    # Evaluate actor with critic in the given state
    #**********************************************
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            # For continuous action spaces, use Multivariate Normal dist. for actions
            dist = MultivariateNormal(action_mean, cov_mat)

            # For 'Single Action' environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            # For discrete action spaces, use Categorical dist. for actions
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # Get Q-values from critic
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

    #*****************************************************
    # Get action distributions (necessary for PPOGraphics)
    #*****************************************************
    def action_distribution(self, state):
        # Reshape state to [batch; N vehicles; features]
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # For continuous action spaces, use Multivariate Normal dist. for actions
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            # For discrete action spaces, use Categorical dist. for actions
            dist = Categorical(action_probs)

        return dist
