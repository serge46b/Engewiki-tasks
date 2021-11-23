#!/usr/bin/env python

import copy
import math
import statistics
from datetime import datetime
import random
from sklearn.cluster import AgglomerativeClustering as clustering_algorithm
# from sklearn.cluster import DBSCAN as clustering_algorithm
# from sklearn.cluster import k_means
import cv2
from PIL import Image
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

import navigator

import glob

from gym_duckietown.envs import DuckietownEnv

# env = gym.make("Duckietown-udem1-v0")
env = DuckietownEnv(map_name="TTIC_large_loop")
env.reset()
env.render()

graph = navigator.Graph("duckietown map graph exemple.json")
navi = navigator.Navi(graph)

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

    mask = cv2.inRange(img_right, (22, 75, 82), (255, 255, 255))
    cv2.imshow("mask", mask)
    left_mask = cv2.inRange(img_left, (20, 20, 70), (50, 50, 170))
    cv2.imshow("left mask", left_mask)
    red_mask = cv2.inRange(img_right, (20, 20, 70), (50, 50, 170))
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


observ = np.uint8(np.zeros((480, 640, 3)))


def update(dt):
    global counter, prev_step, observ, flag, dataset_done
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

    env.step(action)

    # print(round(env.cur_pos[0], 2), round(env.cur_pos[2], 2))
    # print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))
    img = cv2.cvtColor(observ, cv2.COLOR_BGR2RGB)
    # cv2.imshow("original", img)
    cutout_img = np.zeros_like(img)
    # print(cutout_img.shape)
    """
    cv2.line(cutout_img, (75, 210), (565, 210), (255, 255, 255), 4)
    cv2.line(cutout_img, (0, 350), (75, 210), (255, 255, 255), 4)
    cv2.line(cutout_img, (640, 350), (565, 210), (255, 255, 255), 4)"""

    cv2.line(cutout_img, (0, 210), (640, 210), (255, 255, 255), 4)
    cv2.floodFill(cutout_img, None, (639, 479), (255, 255, 255))

    # cv2.imshow("cutout", cutout_img)

    # white_mask = cv2.inRange(out, (22, 75, 82), (255, 255, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)

    # _, th_img1 = cv2.threshold(obs, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold 1", th_img1)

    _, th_img = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    # th_img1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    # cv2.imshow("threshold", th_img)

    # cutted_image = np.zeros_like(img)
    cutout_img = cv2.cvtColor(cutout_img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("cutout", cutout_img)
    # cv2.imshow("threshold", th_img)

    cut_image = cv2.bitwise_and(th_img, cutout_img)
    # cut_image = th_img
    cv2.imshow("cut threshold", cut_image)

    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(cut_image, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("opening", opening_img)

    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(opening_img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 200

    # your answer image
    filtered_img = np.zeros(output.shape)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered_img[output == i + 1] = 255

    # cv2.imshow("filtered", filtered_img)

    filtered_img = np.uint8(filtered_img * 255)

    contours, hierarchy = cv2.findContours(filtered_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print(len(contours), final_mask.shape)
    final_mask = np.zeros_like(img)
    contours_on_orig_img = copy.deepcopy(img)
    contours_colored_img = np.zeros_like(img, dtype=np.uint8)

    cv2.drawContours(contours_on_orig_img, contours, -1, (0, 255, 0), 2)
    centers_on_orig_img = copy.deepcopy(contours_on_orig_img)
    center_cords = []
    contours_data = {}
    min_max_h = [255, 0]
    for c in range(len(contours)):
        m = cv2.moments(contours[c])
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        center_cords.append([cx, cy])
        cv2.circle(centers_on_orig_img, (cx, cy), 1, (0, 0, 255), 3)
        approx = cv2.approxPolyDP(contours[c], 0.008 * cv2.arcLength(contours[c], True), True)
        if not (250 < cv2.contourArea(approx) < 30000):
            continue
        color_rgb = img[cy, cx, ::]
        color_hsv = cv2.cvtColor(np.array([[color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        if color_hsv[2] > 70:
            cv2.drawContours(contours_colored_img, [approx], 0, tuple([int(i) for i in color_rgb]), 3)
            contours_data[c] = {"color": [int(i) for i in color_hsv], "cords": [cx, cy], "approx": approx}

            # contours_data[c] = [color_hsv[1], cx, cy]
            # if min_max_h[1] > color_hsv[1]:
            #     min_max_h[1] = color_hsv[1]
            # if min_max_h[0] < color_hsv[1]:
            #     min_max_h[0] = color_hsv[1]
            # if color_hsv[1] < 125:
            #     text = f"White {color_hsv[1]}"
            #     text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            #     cv2.putText(contours_colored_img, text, (cx - text_size[0] // 2, cy - text_size[1] // 2),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # else:
            #    text = f"yellow {color_hsv[1]}"
            #    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            #     cv2.putText(contours_colored_img, text, (cx - text_size[0] // 2, cy - text_size[1] // 2),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("cnt colored", contours_colored_img)

    sorted_contours = {}
    if len(contours_data) == 1:
        sorted_contours[list(contours_data.keys())[0]] = "white"
    elif len(contours_data) == 2:
        min_s = {"s": contours_data[list(contours_data.keys())[0]]["color"][1], "idx": 0}
        max_s = {"s": 0, "idx": 0}
        for i in contours_data:
            s = contours_data[i]["color"][1]
            if s < min_s["s"]:
                min_s["s"] = s
                min_s["idx"] = i
            elif s > max_s["s"]:
                max_s["s"] = s
                max_s["idx"] = i
        sorted_contours[min_s["idx"]] = "white"
        sorted_contours[max_s["idx"]] = "yellow"
    elif len(contours_data) > 0:
        min_s = {"s": contours_data[list(contours_data.keys())[0]]["color"][1], "idx": 0}
        max_s = {"s": 0, "idx": 0}
        for i in contours_data:
            s = contours_data[i]["color"][1]
            if s < min_s["s"]:
                min_s["s"] = s
                min_s["idx"] = i
            elif s > max_s["s"]:
                max_s["s"] = s
                max_s["idx"] = i
        sorted_contours[min_s["idx"]] = "white"
        sorted_contours[max_s["idx"]] = "yellow"
        itt = 0
        intersections = {}
        for i in contours_data:
            approx = contours_data[i]["approx"]
            # print(approx[0][0], approx[1][0])
            for a in range(len(approx)):
                [x1, y1] = approx[(a == 0) * len(approx) + a - 1][0]
                [x2, y2] = approx[a][0]
                """dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                new_dist = dist * 1
                k = new_dist // dist"""
                k = 1.5  # + (max(y1, y2) == img.shape[0]) * 100
                dx = x2 - x1
                x3 = x2 + int(dx * k)
                y3 = y2 * int(k + 1) - y1
                cv2.line(contours_colored_img, (x1, y1), (x3, y3), (0, 255, 0), 3)

                dx = x1 - x2
                x4 = x1 + int(dx * k)
                y4 = y1 * int(k + 1) - y2
                cv2.line(contours_colored_img, (x1, y1), (x4, y4), (0, 0, 255), 3)

                x11, x12, y11, y12 = x1, x3, y1, y3
                for i2 in contours_data:
                    approx2 = contours_data[i2]["approx"]
                    for a2 in range(len(approx2)):
                        [x21, y21] = approx2[(a2 == 0) * len(approx) + a2 - 1][0]
                        [x22, y22] = approx2[a2][0]
                        maxx1 = max(x11, x12)
                        maxy1 = max(y11, y12)
                        minx1 = min(x11, x12)
                        miny1 = min(y11, y12)
                        maxx2 = max(x21, x22)
                        maxy2 = max(y21, y22)
                        minx2 = min(x21, x22)
                        miny2 = min(y21, y22)

                        if minx1 > maxx2 or maxx1 < minx2 or miny1 > maxy2 or maxy1 < miny2:
                            # print("no intersect")
                            pass
                        else:
                            dx1 = x12 - x11
                            dy1 = y12 - y11
                            dx2 = x22 - x21
                            dy2 = y22 - y21
                            dxx = x11 - x21
                            dyy = y11 - y21
                            div = int(dy2 * dx1 - dx2 * dy1)
                            mul = 0
                            if div == 0:
                                # print("no intersect 1")
                                pass
                            else:
                                if div > 0:
                                    mul = int(dx1 * dyy - dy1 * dxx)
                                    if mul < 0 or mul > div:
                                        # print("no intersect 2")
                                        pass
                                    else:
                                        mul = int(dx2 * dyy - dy2 * dxx)
                                        if mul < 0 or mul > div:
                                            # print("no intersect 3")
                                            pass
                                        else:
                                            # print('intersect')
                                            intersections[itt] = [i, i2, "fw"]
                                            itt += 1
                                else:
                                    mul = -int(dx1 * dyy - dy1 * dxx)
                                    if mul < 0 or mul > -div:
                                        # print("no intersect 4")
                                        pass
                                    else:
                                        mul = -int(dx2 * dyy - dy2 * dxx)
                                        if mul < 0 or mul > -div:
                                            # print("no intersect 5")
                                            pass
                                        else:
                                            # print('intersect')
                                            intersections[itt] = [i, i2, "fw"]
                                            itt += 1
                x21, x22, y21, y22 = x2, x4, y2, y4
                for i2 in contours_data:
                    approx2 = contours_data[i2]["approx"]
                    for a2 in range(len(approx2)):
                        [x11, y11] = approx2[(a2 == 0) * len(approx) + a2 - 1][0]
                        [x12, y12] = approx2[a2][0]
                        maxx1 = max(x11, x12)
                        maxy1 = max(y11, y12)
                        minx1 = min(x11, x12)
                        miny1 = min(y11, y12)
                        maxx2 = max(x21, x22)
                        maxy2 = max(y21, y22)
                        minx2 = min(x21, x22)
                        miny2 = min(y21, y22)

                        if minx1 > maxx2 or maxx1 < minx2 or miny1 > maxy2 or maxy1 < miny2:
                            # print("no intersect")
                            pass
                        else:
                            dx1 = x12 - x11
                            dy1 = y12 - y11
                            dx2 = x22 - x21
                            dy2 = y22 - y21
                            dxx = x11 - x21
                            dyy = y11 - y21
                            div = int(dy2 * dx1 - dx2 * dy1)
                            mul = 0
                            if div == 0:
                                # print("no intersect 1")
                                pass
                            else:
                                if div > 0:
                                    mul = int(dx1 * dyy - dy1 * dxx)
                                    if mul < 0 or mul > div:
                                        # print("no intersect 2")
                                        pass
                                    else:
                                        mul = int(dx2 * dyy - dy2 * dxx)
                                        if mul < 0 or mul > div:
                                            # print("no intersect 3")
                                            pass
                                        else:
                                            # print('intersect')
                                            intersections[itt] = [i, i2, "bw"]
                                            itt += 1
                                else:
                                    mul = -int(dx1 * dyy - dy1 * dxx)
                                    if mul < 0 or mul > -div:
                                        # print("no intersect 4"
                                        pass
                                    else:
                                        mul = -int(dx2 * dyy - dy2 * dxx)
                                        if mul < 0 or mul > -div:
                                            # print("no intersect 5")
                                            pass
                                        else:
                                            # print('intersect')
                                            intersections[itt] = [i, i2, "bw"]
                                            itt += 1
            # print("-------------")
        idx = 0
        for i in intersections:
            if i[0] == max_s["idx"]:
                idx = i
        for i in intersections:
            data = intersections[(i + idx > len(intersections) - 1)]

    cv2.imshow("add lines", contours_colored_img)

    """min_distances = {}
    median_dist = []
    for i in range(len(contours_data)):
        # color = contours_data[list(contours_data.keys())[i]][0]
        # center = contours_data[list(contours_data.keys())[i]][1]
        approx = contours_data[list(contours_data.keys())[i]][2]
        min_distance = img.shape[0]
        orig_min_cords = [0, 0]
        cmp_min_cords = [0, 0]
        min_idx = 0
        for j in range(i + 1, len(contours_data)):
            # color2 = contours_data[list(contours_data.keys())[j]][0]
            # center2 = contours_data[list(contours_data.keys())[j]][1]
            approx2 = contours_data[list(contours_data.keys())[j]][2]
            for a in range(len(approx)):
                [x, y] = approx[a][0]
                for a2 in range(len(approx2)):
                    [x2, y2] = approx2[a2][0]
                    dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
                    if min_distance > dist:
                        min_distance = dist
                        orig_min_cords = [x, y]
                        cmp_min_cords = [x2, y2]
                        min_idx = j
        median_dist.append(min_distance)
        min_distances[list(contours_data.keys())[i]] = {"cmp_point_idx": list(contours_data.keys())[min_idx],
                                                        "min_distance": min_distance,
                                                        "orig_point_cords": orig_min_cords,
                                                        "cmp_point_cords": cmp_min_cords}
    if len(median_dist) != 0:
        median_dist = statistics.mean(median_dist)
    else:
        median_dist = 0
    for i in min_distances:
        color = (0, int(255 * (min_distances[i]["min_distance"] < median_dist)),
                 int(255 * (min_distances[i]["min_distance"] > median_dist)))
        cv2.line(contours_colored_img, min_distances[i]["orig_point_cords"],
                 min_distances[i]["cmp_point_cords"], color, 3)
    cv2.imshow("min distances", contours_colored_img)"""

    """cnt_colored_cords_img = np.uint8(np.zeros((280, 280)))
    for i in contours_data:
        val = contours_data[i][0]
        # val = [100, 100, 200]
        for j in range(val[0], val[0]+25):
            for g in range(val[1], val[1]+25):
                cnt_colored_cords_img[j][g] = val[2]
    cv2.imshow("cnt colored img to coordinates", cnt_colored_cords_img)

    if len(contours_data) >= 2:
        data_for_clusterer = []
        for i in contours_data:
            val = contours_data[i][0]
            # val = [100, 100, 200]
            data_for_clusterer.append(val)

        clusterer = k_means(data_for_clusterer, n_clusters=2)
        #clusterer.fit_predict(data_for_clusterer)
        # cnt_colored_clustered = clusterer.fit_predict(data_for_clusterer)
        cnt_colored_clustered_img = cv2.cvtColor(np.uint8(np.zeros((280, 280, 3))), cv2.COLOR_RGB2HSV)
        contours_colored_img = cv2.cvtColor(contours_colored_img, cv2.COLOR_RGB2HSV)
        print(clusterer[1])
        for i in range(len(contours_data)):
            val = contours_data[list(contours_data.keys())[i]][0]
            # val = [100, 100, 200]
            color = (clusterer[1][i] * 255, 125, val[2])
            for j in range(val[0], val[0] + 25):
                for g in range(val[1], val[1] + 25):
                    cnt_colored_clustered_img[j][g] = color
            m = cv2.moments(contours[list(contours_data.keys())[i]])
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            center_cords.append([cx, cy])
            color = tuple([int(c) for c in color])
            cv2.floodFill(contours_colored_img, None, (cx, cy), color)

        cv2.imshow("cnt colored clustered img", cnt_colored_clustered_img)
        cv2.imshow("filled contours colored clustered img", contours_colored_img)

        print("number of cluster found: {}".format(len(set(clusterer[1]))))

    cnt_colored_cords_img = np.uint8(np.zeros((img.shape[0] + 25, img.shape[1] + 25)))
    for i in contours_data:
        val = contours_data[i]
        # val = [100, 100, 200]
        for j in range(val[1][0], val[1][0] + 25):
            for g in range(val[1][1], val[1][1] + 25):
                cnt_colored_cords_img[g][j] = abs(255 - val[0][1] * 2)
    cv2.imshow("cnt colored img to coordinates", cnt_colored_cords_img)

    if len(contours_data) >= 2:
        data_for_clusterer = []
        for i in contours_data:
            val = [contours_data[i][0][1], contours_data[i][1][0], contours_data[i][1][1]]
            # val = [100, 100, 200]
            data_for_clusterer.append(val)

        # clusterer = k_means(data_for_clusterer, n_clusters=2)
        # clusterer = clustering_algorithm(eps=200, min_samples=1)
        clusterer = clustering_algorithm()
        clusterer.fit_predict(data_for_clusterer)
        # cnt_colored_clustered = clusterer.fit_predict(data_for_clusterer)
        cnt_colored_clustered_img = cv2.cvtColor(np.uint8(np.zeros((img.shape[0] + 25, img.shape[1] + 25, 3))), cv2.COLOR_RGB2HSV)
        contours_colored_img = cv2.cvtColor(contours_colored_img, cv2.COLOR_RGB2HSV)
        print(clusterer.labels_)
        for i in range(len(contours_data)):
            val = [contours_data[list(contours_data.keys())[i]][0][1],
                   contours_data[list(contours_data.keys())[i]][1][0],
                   contours_data[list(contours_data.keys())[i]][1][1]]
            # val = [100, 100, 200]
            color = (clusterer.labels_[i] * 75, 125, 125)
            for j in range(val[1], val[1] + 25):
                for g in range(val[2], val[2] + 25):
                    cnt_colored_clustered_img[g][j] = color
            m = cv2.moments(contours[list(contours_data.keys())[i]])
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            center_cords.append([cx, cy])
            color = tuple([int(c) for c in color])
            cv2.floodFill(contours_colored_img, None, (cx, cy), color)

        cv2.imshow("cnt colored clustered img", cnt_colored_clustered_img)
        cv2.imshow("filled contours colored clustered img", contours_colored_img)

        # print("number of cluster found: {}".format(len(set(clusterer[1]))))"""

    # print('cluster for each point: ', clusterer.labels_)
    # contours_colored_img_norm = np.zeros_like(contours_colored_img)
    # cv2.normalize(contours_colored_img, contours_colored_img_norm, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('cnt colored normalized', contours_colored_img_norm)

    if len(center_cords) > 1:
        min_dist = math.sqrt(
            (center_cords[1][0] - center_cords[0][0]) ** 2 + (center_cords[1][1] - center_cords[0][1]) ** 2)
        for i in range(1, len(center_cords)):
            distX = center_cords[i][0] - center_cords[i - 1][0]
            distY = center_cords[i][1] - center_cords[i - 1][1]
            dist = math.sqrt(distX ** 2 + distY ** 2)
            if dist < min_dist:
                min_dist = dist
        if min_dist > 25:
            print("allowed", counter // 20)
            cv2.drawContours(final_mask, contours, -1, (1, 1, 1), 2)
            for i in range(len(center_cords)):
                fillX = center_cords[i][0]
                fillY = center_cords[i][1]
                cv2.floodFill(final_mask, None, (fillX, fillY), (1, 1, 1))
            counter += 1
            if counter % 20 == 0:
                cv2.imwrite("images/img." + str(counter // 20) + ".png", img)
                cv2.imwrite("masks/img." + str(counter // 20) + ".png", final_mask)
        else:
            print("denied", counter // 20)

    # cv2.imshow('contours', img)  # вывод обработанного кадра в окно
    cv2.imshow("contours and centers", centers_on_orig_img)

    # cv2.imshow("final mask", final_mask)
    if key_handler[key.F]:
        action = pd_driver(observ, env.unwrapped)
    observ, reward, done, info = env.step(action)

    # white_mask = cv2.inRange(cutted_image, (10, 29, 77), (101, 84, 171))
    # cv2.imshow("white mask", white_mask)

    if key_handler[key.RETURN]:
        im = Image.fromarray(observ)
        im.save("pic.png")

    if done:
        print("done!")
        flag = 0
        env.reset()
        env.render()

    if counter // 20 > max_dataset_count:
        dataset_done = True

    if dataset_done:
        print("dataset is done!")
        env.close()
        sys.exit(0)

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
