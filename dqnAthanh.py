import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('Atlantis-ram-v4')

model = DQN(MlpPolicy, env, verbose=3)
model.learn(total_timesteps=1000000, log_interval=1)

observation = env.reset()
total_reward = 0
for i in range(18000):
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)
    env.render()
    total_reward += reward
    if done:
        break

print(total_reward)