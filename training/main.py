import os
import warnings

from training.train_and_eval import TrainAndEvaluate
from utils.load import load_env, load_agent
from utils.plot import plot_rewards, plot_fps, plot_returns, plot_episode_lenghts

warnings.simplefilter(action='ignore', category=UserWarning)
PATH = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    # Select environment: env or env_grid (for conv network only!)
    env_config = 'env'
    # Select agent: dqn_agent or ppo_agent
    agent_config = 'ppo_agent'
    # Select neural network
    model_config = 'social_attention_2h'

    ### Train agent ###
    env = load_env('env/' + env_config + '.json')
    agent = load_agent(model_config, agent_config, env)
    train = TrainAndEvaluate(env, agent, num_episodes=10, display_env=False, display_agent=False)
    print(f"Ready to train {agent} on {env}")

    train.train()

    ### Plot results ###
    os.makedirs(os.path.join(PATH, 'plots'), exist_ok=True)
    plot_rewards(train.num_episodes, train.rewards, PATH)
    plot_fps(train.num_episodes, train.fps, PATH)
    plot_returns(train.num_episodes, train.returns, PATH)
    plot_episode_lenghts(train.num_episodes, train.episode_lengths, PATH)

    ### Test agent ###
    env.configure({"offscreen_rendering": True})
    agent = load_agent(model_config, agent_config, env)
    evaluation = TrainAndEvaluate(env, agent, num_episodes=3, recover=True)
    evaluation.test()
