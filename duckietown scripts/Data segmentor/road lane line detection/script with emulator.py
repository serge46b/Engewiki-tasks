#!/usr/bin/env python

import copy
from datetime import datetime

import cv2
from PIL import Image
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key
import time

from gym_duckietown.envs import DuckietownEnv

env = gym.make("Duckietown-udem1-v0")

env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    global prev_delta
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        prev_delta = 0
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    elif symbol == key.RETURN:
        print('saving screenshot')
        img = env.render('rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite('screenshot ' + str(datetime.now().minute) + " " + str(datetime.now().second) + '.jpg', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

prev_delta = 0
prev_step = 0
flag = 0
next_move = "forward"
edge_line_flag = 0


def update(dt):
    global prev_delta, prev_step, flag, next_move, edge_line_flag
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
        prev_delta = 0
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
        prev_delta = 0
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
        prev_delta = 0
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
        prev_delta = 0
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
        prev_delta = 0
    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    # print(round(env.cur_pos[0], 2), round(env.cur_pos[2], 2))
    # print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    img = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", img)
    cutout_img = np.zeros_like(img)
    # print(cutout_img.shape)
    cv2.line(cutout_img, (150, 210), (490, 210), (255, 255, 255), 4)
    cv2.line(cutout_img, (0, 350), (150, 210), (255, 255, 255), 4)
    cv2.line(cutout_img, (640, 350), (490, 210), (255, 255, 255), 4)
    cv2.floodFill(cutout_img, None, (639, 479), (255, 255, 255))
    #cv2.imshow("cutout", cutout_img)

    out = np.zeros_like(img)
    out[cutout_img == (255, 255, 255)] = img[cutout_img == 255]

    # white_mask = cv2.inRange(out, (22, 75, 82), (255, 255, 255))
    white_mask = cv2.inRange(out, (10, 29, 77), (101, 84, 171))
    cv2.imshow("white mask", white_mask)

    if key_handler[key.RETURN]:
        im = Image.fromarray(obs)
        im.save("pic.png")

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
