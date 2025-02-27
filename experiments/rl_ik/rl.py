import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import Video
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.rl_ik.env import (RandomTrajectoryEnv, RoboticArmEnv,
                                   generate_smooth_trajectories)


class EvalVideoCallback(BaseCallback):

    def __init__(self, eval_env, train_env, check_freq: int = 100, n_eval_episodes: int = 5, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.eval_env = eval_env
        self.train_env = train_env
        self.n_eval_episodes = n_eval_episodes

        # Generate single fixed trajectory for video
        self.eval_trajectory = generate_smooth_trajectories(start_positions=np.array([[0.0, 0.0]]),
                                                            end_positions=np.array([[2.0, 1.0]]),
                                                            duration_sec=3.0,
                                                            dt=self.eval_env.dt,
                                                            noise_std=0.0,
                                                            curviness=0.5)[0]

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Sync normalization stats
            # self.eval_env.obs_rms = deepcopy(self.train_env.obs_rms)
            # self.eval_env.ret_rms = deepcopy(self.train_env.ret_rms)

            # Evaluate policy
            rewards, steps = [], []
            for _ in range(self.n_eval_episodes):
                episode_rewards = episode_steps = 0
                res = self.eval_env.reset()
                obs, _info = res if isinstance(res, tuple) else (res, {})
                done = False

                while not done and episode_steps < 1000:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated = self.eval_env.step(action)[:4]
                    episode_rewards += reward
                    episode_steps += 1
                    done = terminated or truncated

                rewards.append(episode_rewards)
                steps.append(episode_steps)

            if not rewards or not steps:
                self.logger.warn("No evaluation data collected")
                return True

            mean_reward = np.mean(rewards) / np.mean(steps)
            frames = self._record_video()

            # Log results
            self.logger.record("eval/mean_reward_per_step", mean_reward)
            self.logger.record("eval/video", Video(frames, fps=1 / self.eval_env.dt), exclude=("stdout", "json", "csv"))
            self.logger.dump(self.n_calls)

        return True

    def _record_video(self):
        frames = []
        res = self.eval_env.reset()
        obs, _ = res if isinstance(res, tuple) else (res, {})

        done = False
        while not done:
            frame = self.eval_env.render()
            if len(frame.shape) == 3:
                frame = frame[None, ...]
            frames.append(frame)

            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated

        if not frames:
            self.logger.warn("No frames were recorded during video capture")
            return None

        video = np.stack(frames)
        return np.transpose(video, (1, 0, 4, 2, 3))


def train(continue_training: bool = False):

    def make_env():
        return RandomTrajectoryEnv(RoboticArmEnv(steps_ahead=5))

    # train_env = make_env()
    eval_env = make_env()
    train_env = DummyVecEnv([make_env])
    # eval_env = DummyVecEnv([make_env])
    # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    # Initialize PPO agent
    if continue_training:
        model = PPO.load("model_checkpoints/final_model", env=train_env)
    else:
        model = PPO("MlpPolicy",
                    train_env,
                    verbose=1,
                    tensorboard_log="./tensorboard_logs/",
                    learning_rate=1e-4,
                    n_steps=2048 * 4,
                    batch_size=64 * 4,
                    n_epochs=4,
                    gamma=0.98,
                    gae_lambda=0.9,
                    clip_range=0.1,
                    normalize_advantage=True,
                    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                    ent_coef=0.01)

    # Setup callbacks and train
    callbacks = [
        EvalVideoCallback(eval_env, train_env, check_freq=500, n_eval_episodes=500),
        CheckpointCallback(save_freq=10000, save_path="./model_checkpoints/", name_prefix="rl_model")
    ]

    model.learn(total_timesteps=5_000_000, callback=callbacks, progress_bar=True)
    model.save("model_checkpoints/final_model")
    return model


if __name__ == "__main__":
    train(continue_training=True)
