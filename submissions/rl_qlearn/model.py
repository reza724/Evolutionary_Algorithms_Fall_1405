"""RL sample: tabular Q-learning that learns in bench.TradingEnv itself, no labels.

State  = (ret_1 > 0, ret_24 > 0, current position): 12 states. Actions = short / flat / long.
Reward = the env's step return after fees, so holding, flipping and fee costs are all learned from the market path.
"""
import numpy as np

import bench

ACTIONS = np.array([-1.0, 0.0, 1.0])


def state(obs):
    pos = int(np.sign(obs.position)) + 1                            # 0 short, 1 flat, 2 long
    return (int(obs["ret_1"] > 0) * 2 + int(obs["ret_24"] > 0)) * 3 + pos


class QLearn(bench.Model):
    model_type, author = "rl", "sample"
    lookback = 1

    def fit(self, train, episodes=300, steps=2000, alpha=0.05, gamma=0.99, eps=0.1):
        env, rng = bench.TradingEnv(train, self.lookback), np.random.default_rng(0)
        self.q = np.zeros((12, len(ACTIONS)))
        for _ in range(episodes):                                   # each episode: a random 2000-bar stretch of train
            obs, done = env.reset(start=int(rng.integers(0, len(train) - steps - 1)), steps=steps), False
            while not done:
                s = state(obs)
                a = int(rng.integers(len(ACTIONS))) if rng.random() < eps else int(self.q[s].argmax())
                obs, reward, done = env.step(ACTIONS[a])
                target = reward * 100 + (0 if done else gamma * self.q[state(obs)].max())   # reward in %
                self.q[s, a] += alpha * (target - self.q[s, a])
        print(f"rl_qlearn: {episodes} episodes x {steps} bars, greedy action per state {ACTIONS[self.q.argmax(1)].tolist()}")

    def act(self, obs):
        return ACTIONS[self.q[state(obs)].argmax()]
