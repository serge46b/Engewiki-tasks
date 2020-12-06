#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import copy

import cv2
from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

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


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

prev_delta = 0


def update(dt):
    global prev_delta
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
    print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    img = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    img = img[250:, 300:, :]
    mask = cv2.inRange(img, (22, 75, 82), (255, 255, 255))
    cv2.imshow("mask", mask)

    edges = cv2.Canny(mask, 50, 150, apertureSize=5)
    cv2.imshow("edges", edges)
    lines = cv2.HoughLines(edges, 4, np.pi / 180, 200)
    lines_img = copy.deepcopy(img)
    edge_line_img = copy.deepcopy(img)
    edge_line = []
    if lines is not None:
        for line in range(len(lines)):
            rho, theta = lines[line][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            if line == 0:
                cv2.line(edge_line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                edge_line.append([(x1, y1), (x2, y2)])
            elif line == len(lines) - 1:
                cv2.line(edge_line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                edge_line.append([(x1, y1), (x2, y2)])
            lines_img = cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    sample_lines = [[(340, 230), (145, 0)], [(340, 120), (195, 0)]]
    cv2.line(edge_line_img, sample_lines[0][0], sample_lines[0][1], (0, 0, 255), 2)
    cv2.line(edge_line_img, sample_lines[1][0], sample_lines[1][1], (0, 0, 255), 2)
    cv2.imshow("edge lines", edge_line_img)
    cv2.imshow("all lines", lines_img)
    delta = 0
    k = 0.02
    if edge_line:
        ex1 = sample_lines[0][0][0] - edge_line[0][0][0]
        ex2 = sample_lines[0][1][0] - edge_line[0][1][0]
        ey1 = sample_lines[0][0][1] - edge_line[0][0][1]
        ey2 = sample_lines[0][1][1] - edge_line[0][1][1]

        delta = (ex1 + ex2) - (ey1 + ey2)

    else:
        delta = -100
    print(delta)

    if key_handler[key.F]:
        #lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        #angle_from_straight_in_rads = lane_pose.angle_rad
        #kd = 150
        p = 0.5
        delta = delta * k
        #delta = k * (0 - angle_from_straight_in_rads)
        #delta_d = kd * (prev_delta - (0 - angle_from_straight_in_rads))
        #delta_d = kd * (prev_delta - delta)
        #delta += delta_d
        #kp = 0.1
        #p += kp * delta_d
        action += np.array([p, delta])
        prev_delta = delta
        print(delta)
        env.step(action)

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
