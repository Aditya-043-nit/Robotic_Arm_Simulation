
"""
train.py

Trains a PPO agent on KinovaArmEnv using MockArmBackend by default.
Swapping to RealArmBackend later means changing make_env() only --
nothing else in this file needs to change, because arm_env.py never
knows which backend it's holding.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from arm_env import KinovaArmEnv
from arm_backend import MockArmBackend


def make_env():
    """
    Factory function -- SB3's make_vec_env calls this once per parallel
    environment copy, so each one gets its own fresh backend instance.
    Wrapping in Monitor lets SB3 track episode reward/length automatically,
    which is what feeds the default TensorBoard reward curve.
    """
    def _init():
        backend = MockArmBackend(
            joint_names=KinovaArmEnv.JOINT_NAMES,
            joint_limits_lower=KinovaArmEnv.JOINT_LOWER,
            joint_limits_upper=KinovaArmEnv.JOINT_UPPER,
        )
        env = KinovaArmEnv(backend)
        return Monitor(env)
    return _init


class RewardBreakdownCallback(BaseCallback):
    """
    Custom callback that logs each reward term separately to TensorBoard,
    using the 'reward_breakdown' dict we added to info in arm_env.py.
    This is exactly the debugging tool mentioned earlier -- once training
    is running, you can watch each term's curve individually in
    TensorBoard instead of just the total reward.
    """
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            breakdown = info.get('reward_breakdown')
            if breakdown:
                for key, value in breakdown.items():
                    if key != 'total':
                        self.logger.record_mean(f'reward_terms/{key}', value)
        return True


def main():
    n_envs = 4  # cheap with the mock backend; would likely be 1 with RealArmBackend

    print(f"Creating {n_envs} parallel environments...")
    vec_env = make_vec_env(make_env(), n_envs=n_envs)

    print("Validating environment against Gymnasium API...")
    check_env(KinovaArmEnv(MockArmBackend(
        joint_names=KinovaArmEnv.JOINT_NAMES,
        joint_limits_lower=KinovaArmEnv.JOINT_LOWER,
        joint_limits_upper=KinovaArmEnv.JOINT_UPPER,
    )), warn=True)
    print("Environment OK.")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1),
        save_path='./checkpoints/',
        name_prefix='kinova_ppo_mock'
    )
    reward_cb = RewardBreakdownCallback()

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,       # steps collected per env before each PPO update
        batch_size=64,
        n_epochs=10,        # gradient passes over each collected batch
        gamma=0.99,         # discount factor -- how much future reward matters
        gae_lambda=0.95,    # advantage estimation smoothing
        clip_range=0.2,     # PPO's core stability mechanism
        ent_coef=0.01,      # entropy bonus -- encourages continued exploration
        tensorboard_log="./tb_logs/"
    )

    print("Training started...")
    model.learn(
        total_timesteps=200_000,   # start smaller with mock backend to sanity-check learning
        callback=[checkpoint_cb, reward_cb],
        progress_bar=True
    )

    model.save("kinova_ppo_mock_final")
    print("Done. Model saved as kinova_ppo_mock_final.zip")
    vec_env.close()


if __name__ == '__main__':
    main()

