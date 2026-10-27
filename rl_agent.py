import numpy as np

import utils
import TD3


# Research Harness entry point: the training loop of main.py on the Gymnasium API, called by the
# gymnasium-rl eval as train(make_env, total_timesteps, seed). Hyperparameters are main.py's defaults.
# The eval has already seeded Python, NumPy and PyTorch, and it evaluates the returned policy itself.
def train(make_env, total_timesteps, seed):
	env = make_env()
	env.action_space.seed(seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	policy = TD3.TD3(
		state_dim=state_dim,
		action_dim=action_dim,
		max_action=max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2 * max_action,	# Target policy smoothing is scaled wrt the action scale
		noise_clip=0.5 * max_action,
		policy_freq=2,
	)
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=min(int(1e6), total_timesteps))

	start_timesteps = min(int(25e3), total_timesteps // 10)	# main.py's 25e3, shrunk for small budgets
	expl_noise = 0.1
	batch_size = 256

	state, _ = env.reset(seed=seed)
	for t in range(total_timesteps):

		# Select action randomly or according to policy
		if t < start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action. Only a true termination cuts the bootstrap; a time-limit truncation doesn't
		# (what main.py does with env._max_episode_steps)
		next_state, reward, terminated, truncated, _ = env.step(action)
		replay_buffer.add(state, action, next_state, reward, float(terminated))

		state = next_state

		# Train agent after collecting sufficient data
		if t >= start_timesteps:
			policy.train(replay_buffer, batch_size)

		if terminated or truncated:
			state, _ = env.reset()

	env.close()
	return policy.select_action
