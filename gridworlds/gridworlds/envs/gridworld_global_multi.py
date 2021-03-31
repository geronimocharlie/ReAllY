import numpy as np
import time
import gym
from gym import spaces
import random
import cv2

"""
 n x m gridworld
 The agent can move in the grid world.
 There are multiple block position where the agent cannot move to.
 There is one reward position where the agent gets a reward and is done.
 For each other move the agent gets a reward of 0.
 The observation of the agent for each time step is the picture of the current world state including its own position.
"""


class GridWorld_Global_Multi(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, **config):

        self.config = {
            "height": 10,
            "width": 10,
            "block_position": [(1, 1)],
            "reward_position": (2, 3),
            "start_position": (0, 0),
            "reward": 10,
            "step_penalty": -0.5,
            "max_time_steps": 100,
            "player_color": [0.9, 0.1, 0.1],
            "reward_color": [1, 1, 1],
            "block_color": [0.5, 0.5, 0.5],
            "action_dict": {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"},
        }
        self.config.update(config)

        # get correct actions and transitions
        self.action_dict = self.config["action_dict"]
        for k in self.action_dict.keys():
            if self.action_dict[k] == "UP":
                UP = k
            elif self.action_dict[k] == "RIGHT":
                RIGHT = k
            elif self.action_dict[k] == "DOWN":
                DOWN = k
            elif self.action_dict[k] == "LEFT":
                LEFT = k
            else:
                print(f"unsupported action {self.action_dict[k]} with key {k}")
                raise KeyError

        # assert self.config['window_size']%2==1, "the window size must be uneven"
        self.transitions = {UP: (-1, 0), DOWN: (1, 0), RIGHT: (0, 1), LEFT: (0, -1)}

        # get info on grid
        self.height = self.config["height"]
        self.width = self.config["width"]
        self.max_time_steps = self.config["max_time_steps"]
        self.n_states = self.height * self.width
        self.max_blocks = self.n_states - 1
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)
        self.final_reward = self.config["reward"]
        self.step_penalty = self.config["step_penalty"]

        # grid info for renderin
        self.reward_position = np.asarray(self.config["reward_position"])
        self.start_position = np.asarray(self.config["start_position"])
        self.block_position = np.asarray(self.config["block_position"])
        self.position = tuple(self.start_position)
        screen = np.zeros((self.height, self.width, 3))

        for i, block in enumerate(self.block_position):
            if ((block == self.position).all()) or (
                (block == self.reward_position).all()
            ):
                print("Trying to block start position. Refused.")
            else:
                screen[tuple(block)] = self.config["block_color"]
            if i == self.max_blocks:
                print("Maximum numper of blocks reached. Aborting positioning.")
                break
        screen[tuple(self.reward_position)] = self.config["reward_color"]

        self.basic_screen = screen
        self.reset()

        # for some reason gym wants that
        self._seed = random.seed(1234)

    def step(self, action):

        assert self.action_space.contains(action)

        off_x, off_y = self.transitions[action]
        new_position = self.move(off_x, off_y)

        if not (np.any(np.all(new_position == self.block_position, axis=-1))):
            self.position = new_position

        screen = self.basic_screen.copy()
        screen[tuple(self.position)] = self.config["player_color"]
        self.t += 1
        # done if terminal state is reached
        if (self.position == self.reward_position).all():
            self.done = True
            return screen, self.final_reward, self.done, None

        # done if max time steps reached
        if self.t == self.max_time_steps:
            self.done = True

        return screen, self.step_penalty, self.done, None

    def move(self, x_off, y_off):

        x, y = self.position

        # check for borders
        if ((x == 0) & (x_off == -1)) or ((x == self.height - 1) & (x_off == 1)):
            x = x
        else:
            x = x + x_off
        if ((y == 0) & (y_off == -1)) or ((y == self.width - 1) & (y_off == 1)):
            y = y
        else:
            y = y + y_off

        return (x, y)

    def reset(self):
        self.position = tuple(self.start_position)
        self.done = False
        self.t = 1
        screen = self.basic_screen.copy()
        screen[tuple(self.position)] = self.config["player_color"]
        return screen

    def render(self, mode="human", close=False, size_wh=(500, 900), show_view=False):
        screen = self.basic_screen.copy()
        screen[tuple(self.position)] = self.config["player_color"]
        cv2.imshow("GridWorld environment", screen)
        cv2.waitKey(100)
        if close:
            print("closing")
            cv2.destroyAllWindows()
