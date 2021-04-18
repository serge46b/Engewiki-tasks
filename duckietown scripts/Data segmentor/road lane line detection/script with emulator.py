#!/usr/bin/env python

import copy
import math
from datetime import datetime

import cv2
from PIL import Image
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
    elif symbol == key.RETURN:
        print('saving screenshot')
        img = env.render('rgb_array')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite('screenshot ' + str(datetime.now().minute) + " " + str(datetime.now().second) + '.jpg', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

counter = 0


def update(dt):
    global counter, prev_step
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
    img = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    # cv2.imshow("original", img)
    cutout_img = np.zeros_like(img)
    # print(cutout_img.shape)
    cv2.line(cutout_img, (75, 210), (565, 210), (255, 255, 255), 4)
    cv2.line(cutout_img, (0, 350), (75, 210), (255, 255, 255), 4)
    cv2.line(cutout_img, (640, 350), (565, 210), (255, 255, 255), 4)
    cv2.floodFill(cutout_img, None, (639, 479), (255, 255, 255))
    # cv2.imshow("cutout", cutout_img)

    # white_mask = cv2.inRange(out, (22, 75, 82), (255, 255, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)

    # _, th_img1 = cv2.threshold(obs, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold 1", th_img1)

    _, th_img = cv2.threshold(gray, 67, 255, cv2.THRESH_BINARY)
    # th_img1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    cv2.imshow("threshold", th_img)

    # cutted_image = np.zeros_like(img)
    cutout_img = cv2.cvtColor(cutout_img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("cutout", cutout_img)
    # cv2.imshow("threshold", th_img)

    cut_image = cv2.bitwise_and(th_img, cutout_img)
    cv2.imshow("cut threshold", cut_image)

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cut_image, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 800

    # your answer image
    filtered_img = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_img[output == i + 1] = 255

    cv2.imshow("filtered", filtered_img)

    filtered_img = np.uint8(filtered_img * 255)

    contours, hierarchy = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print(len(contours), final_mask.shape)
    final_mask = np.zeros_like(img)
    contours_img = copy.deepcopy(img)
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)
    centers = copy.deepcopy(contours_img)
    center_cords = []
    for c in range(len(contours)):
        m = cv2.moments(contours[c])
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        center_cords.append([cx, cy])
        cv2.circle(centers, (cx, cy), 1, (0, 0, 255), 3)

    if len(center_cords) > 1:
        min_dist = math.sqrt(
            (center_cords[1][0] - center_cords[0][0]) ** 2 + (center_cords[1][1] - center_cords[0][1]) ** 2)
        for i in range(1, len(center_cords)):
            distX = center_cords[i][0] - center_cords[i - 1][0]
            distY = center_cords[i][1] - center_cords[i - 1][1]
            dist = math.sqrt(distX ** 2 + distY ** 2)
            if dist < min_dist:
                min_dist = dist
        if min_dist > 40:
            print("allowed")
            cv2.drawContours(final_mask, contours, -1, (1, 1, 1), 2)
            for i in range(len(center_cords)):
                fillX = center_cords[i][0]
                fillY = center_cords[i][1]
                cv2.floodFill(final_mask, None, (fillX, fillY), (1, 1, 1))
            counter += 1
            if counter % 20 == 0:
                cv2.imwrite("images/img." + str(counter // 20) + ".jpg", img)
                cv2.imwrite("masks/mask." + str(counter // 20) + ".jpg", final_mask)
        else:
            print("denied")

    # cv2.imshow('contours', img)  # вывод обработанного кадра в окно
    cv2.imshow("contours and centers", centers)

    cv2.imshow("final mask", final_mask)


    # white_mask = cv2.inRange(cutted_image, (10, 29, 77), (101, 84, 171))
    # cv2.imshow("white mask", white_mask)

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
