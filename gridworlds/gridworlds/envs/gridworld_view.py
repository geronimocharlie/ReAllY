import numpy as np
import time
import gym
from gym import spaces
import random
import cv2

"""
 n x m gridworld
 The agent can move in the grid world.
 There is one block position where the agent cannot move to.
 There is one reward position where the agent gets a reward and is done.
 For each other move the agent gets a reward of 0.
 The observation of the agent for each time step is a window of size (w,w) and an
 integer indicating the viewing direction of the agent.

"""


class GridWorld_View(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, **config):

        self.config = {
            "height": 3,
            "width": 4,
            "block_position": (1, 1),
            "reward_position": (2, 3),
            "start_position": (0, 0),
            "reward": 10,
            "max_time_steps": 100,
            "player_color": [1, 0, 0],
            "reward_color": [0, 1, 0],
            "block_color": [0, 0, 1],
            "action_dict": {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"},
            "window_size": 3,
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

        assert self.config['window_size']%2==1, "the window size must be uneven"
        self.transitions = {UP: (-1, 0), DOWN: (1, 0), RIGHT: (0, 1), LEFT: (0, -1)}

        # get info on grid
        self.height = self.config["height"]
        self.width = self.config["width"]
        self.window_size = self.config["window_size"]
        self.window_offset = self.window_size//2
        self.max_time_steps = self.config["max_time_steps"]
        self.n_states = self.height * self.width
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.n_states)

        # start state
        self.done = False
        #self.position = self.config["start_position"]
        self.t = 0

        # grid info for renderin
        self.reward_position = tuple(i+self.window_offset for i in self.config["reward_position"])
        self.start_position = tuple(i+self.window_offset for i in self.config["start_position"])
        self.block_position = tuple(i+self.window_offset for i in self.config["block_position"])
        self.position = self.start_position
        screen = np.zeros((self.height, self.width, 3))
        screen = np.pad(screen, ((self.window_offset,self.window_offset), (self.window_offset,self.window_offset), (0,0)), constant_values=0.5)
        screen[self.reward_position] = self.config["reward_color"]
        screen[self.block_position] = self.config["block_color"]
        self.basic_screen = screen

        # for some reason gym wants that
        self._seed = random.seed(1234)
        self.reset()

    def step(self, action):

        assert self.action_space.contains(action)

        off_x, off_y = self.transitions[action]
        new_position = self.move(off_x, off_y)

        if not (new_position == self.block_position):
            self.position = new_position

        screen = self.basic_screen.copy()
        screen[self.position] = self.config["player_color"]
        window = self.get_window(screen)
        self.orientation = action
        # done if terminal state is reached
        if new_position == self.reward_position:
            self.done = True
            return (window, np.expand_dims(np.asarray(self.orientation),0)), self.config["reward"], self.done, None

        # done if max time steps reached
        if self.t == self.max_time_steps:
            self.done = True

        self.t += 1

        return (window, np.expand_dims(np.asarray(self.orientation),0)), 0, self.done, None

    def move(self, x_off, y_off):
        x, y = self.position

        # check for borders
        if ((x == 0+self.window_offset) & (x_off == -1)) or ((x == self.height+self.window_offset - 1) & (x_off == 1)):
            x = x
        else:
            x = x + x_off
        if ((y == 0+self.window_offset) & (y_off == -1)) or ((y == self.width+self.window_offset- 1) & (y_off == 1)):
            y = y
        else:
            y = y + y_off

        return (x, y)

    def view(self, x_off, y_off):
        x, y = self.position
        # check for borders
        if ((x == 0) & (x_off == -1)) or ((x == - 1) & (x_off == 1)):
            x = x
        else:
            x = x + x_off
        if ((y == 0) & (y_off == -1)) or ((y == self.width- 1) & (y_off == 1)):
            y = y
        else:
            y = y + y_off

        return (x, y)



    def reset(self):
        self.position = self.start_position
        self.done = False
        self.t = 0
        screen = self.basic_screen.copy()
        screen[self.position] = self.config["player_color"]
        window = self.get_window(screen)
        # random rientation
        self.orientation = np.random.randint(self.action_space.n)


        return window, np.expand_dims(np.asarray(self.orientation),0)

    def render(self, mode="human", close=False, size_wh=(500,900), show_view=False):
        screen = self.basic_screen.copy()
        screen[self.position] = self.config["player_color"]
        xoff, yoff = self.transitions[self.orientation]
        xv, yv = self.view(xoff, yoff)
        screen[xv,yv] = screen[xv,yv] + [i*0.7 for i in self.config["player_color"]]
        cv2.namedWindow("gridworld", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("gridWorld", *size_wh)
        cv2.imshow("gridworld", screen)
        if show_view:
            cv2.namedWindow("view", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow("view", *size_wh)
        cv2.imshow("view", self.get_window(screen))
        cv2.waitKey(100)


    def get_window(self, screen):
        x,y = self.position
        startx, endx = x-self.window_offset, x+self.window_offset+1
        starty, endy = y-self.window_offset, y+self.window_offset+1
        return screen[startx:endx, starty:endy, :]
