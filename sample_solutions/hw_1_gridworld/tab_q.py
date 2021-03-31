import gridworlds
import gym
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld

"""
This is the sample solution for the tabularQ homework.
In the following we will train an agent on a simple grid world.
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
We are using a custom self build gym like environment
"""


class TabularQ(object):

    def __init__(self, h, w, action_space):

        self.q = np.zeros((h, w, action_space))

    def __call__(self, state):
        # remove empty batch size
        state = np.squeeze(state)
        h, w = state
        output = {}
        # output q vals for each action
        q_vals =  self.q[h,w,:]
        # introduce empty batch size
        q_vals = np.expand_dims(q_vals, axis=0)
        # really requres dict as output
        output['q_values'] = q_vals
        return output

    def get_q(self, state, action):
        # remove empty batch size
        state = np.squeeze(state)
        h, w = state
        return self.q[h,w,action]

    def set_q(self, state, action, val):
        # remove empty batch size
        state = np.squeeze(state)
        h, w = state
        q_vals = self.q.copy()
        q_vals[h, w, action] = val
        self.q = q_vals

    def get_weights(self):
        return self.q.copy()

    def set_weights(self, q_vals):
        self.q = q_vals

    def max_q(self, state):
        # remove empty batch size
        state = np.squeeze(state)
        h, w = state
        return np.max(self.q[h,w,:])



def train_step(model, state, action, reward, new_state, not_done, learning_rate, discount):
    old_q = model.get_q(state, action)
    max_q = model.max_q(new_state)
    update = learning_rate * (reward + not_done * (discount * max_q) - old_q)
    new_q = old_q + update
    model.set_q(state, action, new_q)


def show_q(model, epoch, path, action_dict, max_reward=10):
    qs = model.q.copy()
    path = path + '/q_vals_epoch_' + str(epoch) + '.png'

    # create image for each action
    im_l = [np.squeeze(q) for q in np.split(qs, 4, axis=-1)]
    [print(i) for i in im_l]
    print('\n')

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(f'Q-table after {epoch} training epochs')
    v = 0
    for j in range(2):
        for i in range(2):
            im = axes[j,i].imshow(im_l[v],vmin=0, vmax=max_reward)
            axes[j,i].set_title(action_dict[v])
            v += 1

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(path)

if __name__ == '__main__':
    action_dict = {
        0 : 'UP',
        1 : 'RIGHT',
        2 : 'DOWN',
        3 : 'LEFT'
    }

    env_kwargs = {
        'height': 3,
        'width' : 4,
        'action_dict' : action_dict,
        'start_position' : (2,0),
        'reward_position' : (0,3)
    }

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)
    observation_size = env.observation_space.n
    num_actions = env.action_space.n
    model = TabularQ(env_kwargs['height'],env_kwargs['width'], num_actions)
    # initial weights
    weights = model.get_weights()
    state = env.reset()
    model(state)

    model_kwargs = {
        'h' : env.height,
        'w' : env.width,
        'action_space' : num_actions
    }

    kwargs = {
            'model' : TabularQ,
            'environment' : GridWorld,
            'env_kwargs' : env_kwargs,
            'num_parallel' :8,
            'total_steps' :100,
            'action_sampling_type' :'epsilon_greedy',
            'model_kwargs': model_kwargs,
            'input_shape': False, # no input shape needed for getting first weights
            'weights' : weights,
            'num_episodes': 10,
            'epsilon': 0.8,
            'is_tf' : False
        }

    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)

    saving_path = os.getcwd()+'/progress_tabq'
    saving_after = 5

    # prameters for optimization
    buffer_size = 500
    test_steps = 100
    epochs = 10
    sample_size = 500 # training steps per epoch
    # discount
    gamma = 0.95
    learning_rate = 0.2

    # keys needed for tabular q
    optim_keys = ['state', 'action', 'reward', 'state_new', 'not_done']

    # initialize buffer
    manager.initialize_buffer(buffer_size)
    agent = manager.get_agent()

    # initialize progress aggregator
    manager.initialize_aggregator(path=saving_path, saving_after=saving_after, aggregator_keys=['time_steps', 'rewards'])

    # initial testing:
    print('test before training: ')
    manager.test(test_steps, test_episodes=10, do_print=True, evaluation_measure='time_and_reward')

    for e in range(epochs):

        print('collecting experience..')
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        experience_dict = manager.sample(sample_size)

        print('optimizing...')
        for states, actions, rewards, states_new, not_dones in zip(*[experience_dict[k] for k in optim_keys]):
            train_step(agent.model, states, actions, rewards, states_new, not_dones, learning_rate, gamma)

        # set new weights, get optimized agent
        manager.set_agent(agent.model.get_weights())

        # update aggregator
        time_steps, reward_agg = manager.test(test_steps, evaluation_measure='time_and_reward')
        manager.update_aggregator(time_steps=time_steps, rewards=reward_agg)

        if e%saving_after == 0:
            show_q(agent.model, e, saving_path, env_kwargs['action_dict'])

        print(f"epoch ::: {e}    avg env steps ::: {np.mean(time_steps)}    avg reward ::: {np.mean(reward_agg)}" )

    print('done')
    print('testing optimized agent')
    manager.test(test_steps, render=True, evaluation_measure='time_and_reward')
