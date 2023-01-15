import json
import gym

from utils.config import Configurable
from dqn_agent.dqn_agent import DQNAgent
from ppo_agent.ppo_agent import PPOAgent


#******************************************************
# Create RL agent from the model and agent config files
#******************************************************
def create_agent(env, model_config, agent_config):
    if agent_config["type"] == "DQNAgent":
        agent = DQNAgent(env, lr=5e-4, weight_decay=1e-5, agent_config=agent_config, model_config=model_config)
        return agent
    elif agent_config["type"] == "PPOAgent":
        agent = PPOAgent(env, lr_actor=3e-4, lr_critic=1e-3, weight_decay=1e-5, agent_config=agent_config, policy_config=model_config)
        return agent
    else:
        raise ValueError("Unknown agent type!")


#****************************************
# Load neural network model configuration
#****************************************
def load_model_config(config_path):
    with open(config_path) as f:
        model_config = json.loads(f.read())
    if "base_config" in model_config:
        base_config = load_model_config(model_config["base_config"])
        del model_config["base_config"]
        model_config = Configurable.custom_config(base_config, model_config)
    return model_config


#****************************
# Load RL agent configuration
#****************************
def load_agent_config(config_path):
    with open(config_path) as f:
        agent_config = json.loads(f.read())
    return agent_config


#*************************
# Load and create RL agent
#*************************
def load_agent(model_config, agent_config, env):
    if not isinstance(model_config, dict):
        if agent_config == 'dqn_agent':
            model_config = load_model_config('model/' + model_config + '.json')
        elif agent_config == 'ppo_agent':
            actor_config = load_model_config('model/' + 'actor_' + model_config + '.json')
            critic_config = load_model_config('model/' + 'critic_' + model_config + '.json')
            model_config = {
                "actor_network": actor_config,
                "critic_network": critic_config
            }
        else:
            raise ValueError("Unknown agent type!")
    if not isinstance(agent_config, dict):
        agent_config = load_agent_config('agent/' + agent_config + '.json')
    return create_agent(env, model_config, agent_config)


#*****************
# Load environment
#*****************
def load_env(env_config):
    if not isinstance(env_config, dict):
        with open(env_config) as f:
            env_config = json.loads(f.read())
    if env_config.get("import_module", None):
        __import__(env_config["import_module"])

    env = gym.make(env_config['id'])
    env.import_module = env_config.get("import_module", None)

    env.unwrapped.configure(env_config)
    env.reset()

    return env
