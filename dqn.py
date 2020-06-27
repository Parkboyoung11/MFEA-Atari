

import gym
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

env_name = ['Amidar-ram-v4', 'Asterix-ram-v4', 'Asteroids-ram-v4', 'BeamRider-ram-v4', 'Bowling-ram-v4', 'Centipede-ram-v4', 'ChopperCommand-ram-v4', 'Enduro-ram-v4']

for i in range(0, 8):
	name = env_name[i]
	env = gym.make(name)
	env = DummyVecEnv([lambda: env]) 

	# model = DQN(MlpPolicy, env, verbose=1, learning_rate=0.0002, buffer_size=100000, batch_size=32)
	model = DQN(MlpPolicy, env, verbose=1, learning_rate=0.0002)
	model.learn(total_timesteps=25000)
	obs = env.reset()

	# result = 0
	results = []
	for _ in range(0, 100):
		result = 0
		while True:
			action, _states = model.predict(obs)
			obs, rewards, dones, info = env.step(action)
			result += rewards[0]
			if dones: break
		results.append(result)
	print(name)
	print(results)
	print(max(results))
	print('\n')





# print(result)


# import gym

# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.deepq.policies import MlpPolicy
# from stable_baselines import DQN

# env = gym.make('CartPole-v1')

# model = DQN(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("deepq_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("deepq_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()