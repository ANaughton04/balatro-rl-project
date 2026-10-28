import numpy as np
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter

#DirectAdapter runs the pure-Python simulator in-process
env = BalatroGymnasiumEnv(adapter_factory = DirectAdapter)
obs, info = env.reset()

#Fetch the 1D boolean Numpy array of legal actions
mask = env.action_masks()

print('Observation type:', type(obs))
if isinstance(obs, dict):
    print('Observation keys:', list(obs.keys()))
    for k, v in obs.items():
        print(f'  - {k}: shape={getattr(v, 'shape', type(v))}, dtype={getattr(v, 'dtype', '')}')
else:
    print('Observation shape:', getattr(obs, 'shape', type(obs)))

print('\nTotal discrete action space size:', len(mask))
print('Number of legal actions right now:', int(np.sum(mask)))
print('Valid action indices right now:', np.flatnonzero(mask))

#Play one random game using only legal actions
steps = 0
total_reward = 0.0
done = False

while not done:
    #Get the current legal action mask
    mask = env.action_masks()
    valid_actions = np.flatnonzero(mask)

    #Pick a random legal action
    action = int(np.random.choice(valid_actions))

    #Step the envrionment once
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += float(reward)
    done = bool(terminated or truncated)
    steps += 1

print(f'Episode finished in {steps} steps!  Final info: {info}')

#Print info without the giant 500-boolean action_mask array so it's readable
clean_info = {k: v for k, v in info.items() if k != 'action_mask'}
print(f'Final info: {clean_info}')