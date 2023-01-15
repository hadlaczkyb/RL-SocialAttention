import datetime
import json
import logging
import os
import time
import torch
import numpy as np

from pathlib import Path

from training.reward_viewer import RewardViewer
from training.monitor import MonitorV2
from utils.config import serialize

PATH = os.path.dirname(os.path.abspath(__file__))

# TODO: - import logger
logger = logging.getLogger(__name__)


class TrainAndEvaluate(object):
    OUTPUT_FOLDER = 'out'
    SAVED_MODELS_FOLDER = 'saved_models'
    RUN_FOLDER = 'run_{}_{}'
    METADATA_FILE = 'metadata.{}.json'
    LOGGING_FILE = 'logging.{}.log'

    def __init__(self, env, agent, directory=None, run_directory=None, num_episodes=1000, training=True, sim_seed=None,
                 recover=None, display_env=True, display_agent=True, display_rewards=True, close_env=True):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.training = training
        self.sim_seed = sim_seed
        self.close_env = close_env
        self.display_env = display_env

        self.directory = Path(directory or self.default_directory)
        self.run_directory = self.directory / (run_directory or self.default_run_directory)

        self.monitor = MonitorV2(env, self.run_directory, video_callable=(None if self.display_env else False))

        self.episode = 0
        self.agent_evaluation = self
        self.write_metadata()
        self.filtered_agent_stats = 0
        self.best_agent_stats = -np.infty, 0

        # Init arrays for plotting
        self.rewards = torch.zeros(num_episodes)
        self.returns = torch.zeros(num_episodes)
        self.fps = torch.zeros(num_episodes)
        self.episode_lengths = torch.zeros(num_episodes)

        self.recover = recover
        if self.recover:
            self.load_agent_model(self.recover)

        if display_agent:
            try:
                # Render the agent in the environment viewer
                self.env.render()
                self.env.unwrapped.viewer.directory = self.run_directory
            except AttributeError:
                pass

        if display_rewards:
            self.reward_viewer = RewardViewer()

        self.observation = None

    def train(self):
        self.training = True
        self.run_episodes()
        self.close()

    def test(self):
        self.training = False
        if self.display_env:
            self.monitor.video_callable = MonitorV2.always_call_video
        try:
            self.agent.eval()
        except AttributeError:
            pass
        self.run_episodes()
        self.close()

    def run_episodes(self):
        for self.episode in range(self.num_episodes):
            # Run episode
            terminal = False
            self.seed(self.episode)
            self.reset()
            rewards = []
            start_time = time.time()
            # Run until a terminal step is reached
            while not terminal:
                reward, terminal = self.step()
                rewards.append(reward)
                try:
                    if self.env.unwrapped.done:
                        break
                except AttributeError:
                    pass

            # After episodes
            duration = time.time() - start_time
            self.after_all_episodes(self.episode, rewards, duration)
            self.after_some_episodes(self.episode, rewards)

    def step(self):
        # Get action sequence from agent
        actions = self.agent.plan(self.observation)

        # Forward the actions to the env. viewer
        try:
            self.env.unwrapped.viewer.set_agent_action_sequence(actions)
        except AttributeError:
            pass

        # Step the environment
        previous_observation, action = self.observation, actions[0]
        self.observation, reward, terminal, info = self.monitor.step(action)

        # Record
        try:
            self.agent.record(previous_observation, action, reward, self.observation, terminal, info)
        except NotImplementedError:
            pass

        return reward, terminal

    def save_agent_model(self, identifier, do_save=True):
        # Create folder if it doesn't exist
        permanent_folder = self.directory / self.SAVED_MODELS_FOLDER
        os.makedirs(permanent_folder, exist_ok=True)

        episode_path = None
        if do_save:
            episode_path = Path(self.monitor.directory) / "checkpoint-{}.tar".format(identifier)
            try:
                self.agent.save(filename=permanent_folder / "latest.tar")
                episode_path = self.agent.save(filename=episode_path)
                if episode_path:
                    logger.info("Saved {} model to {}".format(self.agent.__class__.__name__, episode_path))
            except NotImplementedError:
                pass
        return episode_path

    def load_agent_model(self, model_path):
        if model_path is True:
            model_path = self.directory / self.SAVED_MODELS_FOLDER / "latest.tar"
        if isinstance(model_path, str):
            model_path = Path(model_path)
            if not model_path.exists():
                model_path = self.directory / self.SAVED_MODELS_FOLDER / model_path
        try:
            model_path = self.agent.load(filename=model_path)
            if model_path:
                logger.info("Loaded {} model from {}".format(self.agent.__class__.__name__, model_path))
        except FileNotFoundError:
            pass
        except NotImplementedError:
            pass

    def after_all_episodes(self, episode, rewards, duration):
        rewards = np.array(rewards)
        gamma = self.agent.config.get("gamma", 1)
        print("Episode {} score: {:.1f}".format(episode, sum(rewards)))

        # Save episode results for plotting
        self.rewards[episode] = sum(rewards)
        self.returns[episode] = sum(r * gamma ** t for t, r in enumerate(rewards))
        self.fps[episode] = len(rewards) / duration
        self.episode_lengths[episode] = len(rewards)

    def after_some_episodes(self, episode, rewards, best_increase=1.1, episodes_window=50):
        if self.monitor.is_episode_selected():
            # Save the model
            if self.training:
                self.save_agent_model(episode)

        if self.training:
            # Save best model so far, averaged on a window
            best_reward, best_episode = self.best_agent_stats
            self.filtered_agent_stats += 1 / episodes_window * (np.sum(rewards) - self.filtered_agent_stats)
            if self.filtered_agent_stats > best_increase * best_reward \
                    and episode >= best_episode + episodes_window:
                self.best_agent_stats = (self.filtered_agent_stats, episode)
                self.save_agent_model("best")

    @property
    def default_directory(self):
        return Path(self.OUTPUT_FOLDER) / self.env.unwrapped.__class__.__name__ / self.agent.__class__.__name__

    @property
    def default_run_directory(self):
        return self.RUN_FOLDER.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), os.getpid())

    def write_metadata(self):
        metadata = dict(env=serialize(self.env), agent=serialize(self.agent))
        file_infix = '{}.{}'.format(self.monitor.monitor_id, os.getpid())
        file = self.run_directory / self.METADATA_FILE.format(file_infix)
        with file.open('w') as f:
            json.dump(metadata, f, sort_keys=True, indent=4)

    def seed(self, episode=0):
        seed = self.sim_seed + episode if self.sim_seed is not None else None
        seed = self.monitor.seed(seed)
        self.agent.seed(seed[0])  # Seed the agent with the main environment seed
        return seed

    def reset(self):
        self.observation = self.monitor.reset()
        self.agent.reset()

    def close(self):
        if self.training:
            self.save_agent_model("final")
        self.monitor.close()
        if self.close_env:
            self.env.close()
