#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import copy
import glob

from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv
import cv2

import random
import navigator

# env = DuckietownEnv(map_name="udem1_empty", domain_rand=False, style="segmentation")
env = DuckietownEnv(map_name="udem1", domain_rand=False, style="photos")
env.reset()
env.render()

graph = navigator.Graph("duckietown map graph exemple.json")
navi = navigator.Navi(graph)

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


def pd_driver(obs, unwrapped_env):
    global prev_delta, prev_step, flag, next_move, edge_line_flag
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

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

    img_right = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    img_left = copy.deepcopy(img_right)
    img_right = img_right[250:, 300:, :]
    img_left = img_left[160:300, :340, :]

    mask = cv2.inRange(img_right, (250, 250, 250), (255, 255, 255))
    cv2.imshow("mask", mask)
    left_mask = cv2.inRange(img_left, (0, 0, 250), (5, 5, 255))
    cv2.imshow("left mask", left_mask)
    red_mask = cv2.inRange(img_right, (0, 0, 250), (5, 5, 255))
    cv2.imshow("red mask", red_mask)

    edges = cv2.Canny(mask, 50, 150, apertureSize=5)
    cv2.imshow("edges", edges)
    lines = cv2.HoughLines(edges, 4, np.pi / 180, 200)
    lines_img = copy.deepcopy(img_right)
    edge_line_img = copy.deepcopy(img_right)
    edge_line = []

    edges = cv2.Canny(red_mask, 50, 150, apertureSize=5)
    cv2.imshow("red edges", edges)
    red_lines = cv2.HoughLines(edges, 4, np.pi / 180, 200)
    red_lines_img = copy.deepcopy(img_right)
    red_edge_line_img = copy.deepcopy(img_right)
    red_edge_line = []

    edges = cv2.Canny(left_mask, 50, 150, apertureSize=5)
    cv2.imshow("left edges", edges)
    left_lines = cv2.HoughLines(edges, 4, np.pi / 180, 200)
    left_lines_img = copy.deepcopy(img_left)
    left_edge_line_img = copy.deepcopy(img_left)
    left_edge_line = []

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
            # elif line == len(lines) - 1:
            #    cv2.line(edge_line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #    edge_line.append([(x1, y1), (x2, y2)])
            lines_img = cv2.line(lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if red_lines is not None:
        for line in range(len(red_lines)):
            rho, theta = red_lines[line][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            if line == 0:
                cv2.line(red_edge_line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                red_edge_line.append([(x1, y1), (x2, y2)])
            # elif line == len(lines) - 1:
            #    cv2.line(edge_line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #    edge_line.append([(x1, y1), (x2, y2)])
            red_lines_img = cv2.line(red_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if left_lines is not None:
        for line in range(len(left_lines)):
            rho, theta = left_lines[line][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            if line == 0:
                cv2.line(left_edge_line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                left_edge_line.append([(x1, y1), (x2, y2)])
            # elif line == len(lines) - 1:
            #    cv2.line(edge_line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #    edge_line.append([(x1, y1), (x2, y2)])
            left_lines_img = cv2.line(left_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    sample_lines = [(340, 200), (165, 0)]
    red_sample_lines = [(0, 115), (340, 115)]
    left_sample_lines = [(0, 115), (340, 115)]
    cv2.line(edge_line_img, sample_lines[0], sample_lines[1], (0, 0, 255), 2)
    cv2.line(red_edge_line_img, red_sample_lines[0], red_sample_lines[1], (0, 255, 255), 2)
    cv2.line(left_edge_line_img, left_sample_lines[0], left_sample_lines[1], (0, 255, 255), 2)
    # cv2.line(edge_line_img, sample_lines[1][0], sample_lines[1][1], (0, 0, 255), 2)
    cv2.imshow("edge lines", edge_line_img)
    cv2.imshow("all lines", lines_img)
    # cv2.imshow("red edge lines", red_edge_line_img)
    # cv2.imshow("red all lines", red_lines_img)
    # cv2.imshow("left edge lines", left_edge_line_img)
    # cv2.imshow("left all lines", left_lines_img)

    delta = 0
    k = 0.015
    if edge_line:
        ex1 = sample_lines[0][0] - edge_line[0][0][0]
        ex2 = sample_lines[1][0] - edge_line[0][1][0]
        ey1 = sample_lines[0][1] - edge_line[0][0][1]
        ey2 = sample_lines[1][1] - edge_line[0][1][1]

        delta = ((ex1 + ex2) - (ey1 + ey2))
    else:
        delta = -180

    if red_edge_line and not flag:
        rex1 = red_sample_lines[0][0] - (1000 + red_edge_line[0][0][0])
        rex2 = red_sample_lines[1][0] - (1340 - red_edge_line[0][1][0])
        rey1 = red_sample_lines[0][1] - red_edge_line[0][0][1]
        rey2 = red_sample_lines[1][1] - red_edge_line[0][1][1]
        # print(rex1, rex2, rey1, rey2)
        # print(red_edge_line[0][0][0], red_edge_line[0][1][0], red_edge_line[0][0][1], red_edge_line[0][1][1])
        if -75 < rex1 < 75 and -75 < rex2 < 75 and -75 < rey1 < 75 and -75 < rey2 < 75:
            flag = 1
            next_move = "error"
            while next_move == "error":
                x, y = round(unwrapped_env.cur_pos[0], 2), round(unwrapped_env.cur_pos[2], 2)
                angle = round(unwrapped_env.cur_angle * 57)
                print(x, y, angle)
                now_id = graph.get_point_id(x, y, angle)
                print(now_id)
                neighbours = graph.get_neighbours_id(now_id)
                print(neighbours)
                chosen_neighbour = random.choice(neighbours)
                print(chosen_neighbour)
                direction = navi.get_direction_from_point(now_id, chosen_neighbour, angle)
                print(direction)
                next_move = direction
            prev_step = unwrapped_env.step_count
    # print(flag)
    if flag:
        if next_move == "forward":
            if left_edge_line:
                rex1 = left_sample_lines[0][0] - (1000 + left_edge_line[0][0][0])
                rex2 = left_sample_lines[1][0] - (1340 - left_edge_line[0][1][0])
                rey1 = left_sample_lines[0][1] - left_edge_line[0][0][1]
                rey2 = left_sample_lines[1][1] - left_edge_line[0][1][1]
                # print(rex1, rex2, rey1, rey2)
                if -50 < rex1 < 50 and -50 < rex2 < 50 and -50 < rey1 < 50 and -50 < rey2 < 50:
                    flag = 0
        else:
            if edge_line and edge_line_flag == 1:
                prev_step = unwrapped_env.step_count
                flag = 0
                edge_line_flag = 0
                print("ok")
            elif not edge_line:
                if unwrapped_env.step_count - prev_step > 50:
                    edge_line_flag = 1
            print(edge_line_flag, len(edge_line))
    elif prev_step != 0:
        if unwrapped_env.step_count - prev_step > 70:
            prev_step = 0
            flag = 0

        # lane_pose = env.get_lane_pos2(env.cur_pos, env.cur_angle)
        # angle_from_straight_in_rads = lane_pose.angle_rad
        # kd = 1
    p = 0.5
    if flag:
        if next_move == "left":
            delta = 80
        # elif next_move == "right":
        #    delta = -140
        elif next_move == "forward":
            delta = 0

    delta = delta * k
    # delta_d = kd * (prev_delta - (0 - angle_from_straight_in_rads))
    # delta_d = kd * (prev_delta - delta)
    # delta += delta_d
    # kp = 0.1
    # p += kp * delta_d
    action += np.array([p, delta])
    # prev_delta = delta
    # print(delta)
    return action


max_dataset_count = 1005
max_num = 0
counter = 0
dataset_done = False
for name in glob.glob("images/*.png"):
    num = ""
    for s in name:
        if s.isnumeric():
            num += s
    num = int(num)
    if max_num < num:
        max_num = num
if max_num > max_dataset_count:
    dataset_done = True
else:
    counter = max_num * 20
obs = np.uint8(np.zeros((480, 640, 3)))
prev_delta = 0
prev_step = 0
flag = 0
next_move = "forward"
edge_line_flag = 0


def update(dt):
    global counter, prev_step, obs, flag, dataset_done
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    wheel_distance = 0.102
    min_rad = 0.08

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

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

    # seg_obs = env.render_obs(segment=True)
    seg_obs = env.render_obs(segment=True)

    img = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    seg_img = cv2.cvtColor(seg_obs, cv2.COLOR_BGR2RGB)

    border_lines_mask = cv2.inRange(seg_img, (250, 250, 250), (255, 255, 255))
    middle_lines_mask = cv2.inRange(seg_img, (0, 250, 250), (5, 255, 255))
    crossroad_lines_mask = cv2.inRange(seg_img, (0, 0, 250), (5, 5, 255))

    brd_ln_fn_mask = np.zeros_like(seg_img)
    brd_ln_fn_mask[:] = (255, 255, 0)
    brd_ln_fn_mask = cv2.bitwise_and(brd_ln_fn_mask, brd_ln_fn_mask, mask=border_lines_mask)
    cv2.imshow("brd_with_mask", cv2.add(img, brd_ln_fn_mask))
    mdl_ln_fn_mask = np.zeros_like(seg_img)
    mdl_ln_fn_mask[:] = (0, 125, 255)
    mdl_ln_fn_mask = cv2.bitwise_and(mdl_ln_fn_mask, mdl_ln_fn_mask, mask=middle_lines_mask)
    crsrd_ln_fn_mask = np.zeros_like(seg_img)
    crsrd_ln_fn_mask[:] = (255, 0, 255)
    crsrd_ln_fn_mask = cv2.bitwise_and(crsrd_ln_fn_mask, crsrd_ln_fn_mask, mask=crossroad_lines_mask)

    final_mask = cv2.add(cv2.add(brd_ln_fn_mask, mdl_ln_fn_mask), crsrd_ln_fn_mask)

    cv2.imshow("final mask", final_mask)

    cv2.imshow("image", img)
    cv2.imshow("segment", seg_img)

    if key_handler[key.F]:
        action = pd_driver(seg_obs, env.unwrapped)
    obs, reward, done, info = env.step(action)

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()


env.close()
