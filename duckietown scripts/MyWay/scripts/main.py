#!/usr/bin/env python

import json
import copy

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

key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


#@env.unwrapped.window.event
class Driver:
    def __init__(self, my_env, class_navi):
        self.__prev_delta = 0
        self.__prev_step = 0
        self.__flag = 0
        self.__next_move = "forward"
        self.__edge_line_flag = 0
        self.__env = my_env
        self.__navi = class_navi

    def update(self, dt):
        wheel_distance = 0.102
        min_rad = 0.08

        action = np.array([0.0, 0.0])

        if key_handler[key.UP]:
            action += np.array([0.44, 0.0])
            self.__prev_delta = 0
        if key_handler[key.DOWN]:
            action -= np.array([0.44, 0])
            self.__prev_delta = 0
        if key_handler[key.LEFT]:
            action += np.array([0, 1])
            self.__prev_delta = 0
        if key_handler[key.RIGHT]:
            action -= np.array([0, 1])
            self.__prev_delta = 0
        if key_handler[key.SPACE]:
            action = np.array([0, 0])
            self.__prev_delta = 0
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

        obs, reward, done, info = self.__env.step(action)
        # print("step_count = %s, reward=%.3f" % (self.__env.unwrapped.step_count, reward))
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
        cv2.imshow("red edge lines", red_edge_line_img)
        cv2.imshow("red all lines", red_lines_img)
        cv2.imshow("left edge lines", left_edge_line_img)
        cv2.imshow("left all lines", left_lines_img)

        if key_handler[key.F]:
            delta = 0
            k = 0.015
            if edge_line:
                ex1 = sample_lines[0][0] - edge_line[0][0][0]
                ex2 = sample_lines[1][0] - edge_line[0][1][0]
                ey1 = sample_lines[0][1] - edge_line[0][0][1]
                ey2 = sample_lines[1][1] - edge_line[0][1][1]

                delta = ((ex1 + ex2) - (ey1 + ey2))
            else:
                delta = -175

            if red_edge_line and not self.__flag:
                rex1 = red_sample_lines[0][0] - (1000 + red_edge_line[0][0][0])
                rex2 = red_sample_lines[1][0] - (1340 - red_edge_line[0][1][0])
                rey1 = red_sample_lines[0][1] - red_edge_line[0][0][1]
                rey2 = red_sample_lines[1][1] - red_edge_line[0][1][1]
                # print(rex1, rex2, rey1, rey2)
                # print(red_edge_line[0][0][0], red_edge_line[0][1][0], red_edge_line[0][0][1], red_edge_line[0][1][1])
                if -75 < rex1 < 75 and -75 < rex2 < 75 and -75 < rey1 < 75 and -75 < rey2 < 75:
                    self.__flag = 1
                    print(round(self.__env.cur_pos[0], 2), round(self.__env.cur_pos[2], 2))
                    self.__next_move = self.__navi.next_step(round(self.__env.cur_pos[0], 2), round(self.__env.cur_pos[2], 2))
                    print(self.__next_move)
                    self.__prev_step = self.__env.unwrapped.step_count
            # print(self.__flag)
            p = 0.5
            if self.__flag:
                if self.__next_move == "forward":
                    if left_edge_line:
                        rex1 = left_sample_lines[0][0] - (1000 + left_edge_line[0][0][0])
                        rex2 = left_sample_lines[1][0] - (1340 - left_edge_line[0][1][0])
                        rey1 = left_sample_lines[0][1] - left_edge_line[0][0][1]
                        rey2 = left_sample_lines[1][1] - left_edge_line[0][1][1]
                        # print(rex1, rex2, rey1, rey2)
                        if -50 < rex1 < 50 and -50 < rex2 < 50 and -50 < rey1 < 50 and -50 < rey2 < 50:
                            self.__flag = 0
                elif self.__next_move == "arrived":
                    p = 0
                    if key_handler[key.C]:
                        self.__flag = 0
                        p = 0.5
                else:
                    if edge_line and self.__edge_line_flag == 1:
                        self.__prev_step = self.__env.unwrapped.step_count
                        self.__flag = 0
                        self.edge_line_flag = 0
                        print("ok")
                    elif not edge_line:
                        if self.__env.unwrapped.step_count - self.__prev_step > 50:
                            self.__edge_line_flag = 1
                    print(self.__edge_line_flag, len(edge_line))
            elif self.__prev_step != 0:
                if self.__env.unwrapped.step_count - self.__prev_step > 70:
                    self.__prev_step = 0
                    self.__flag = 0
            # lane_pose = self.__env.get_lane_pos2(self.__env.cur_pos, self.__env.cur_angle)
            # angle_from_straight_in_rads = lane_pose.angle_rad
            # kd = 1
            if self.__flag:
                if self.__next_move == "left":
                    delta = 70
                # elif self.__next_move == "right":
                #    delta = -140
                elif self.__next_move == "forward":
                   delta = 0
            delta = delta * k
            # delta_d = kd * (self.__prev_delta - (0 - angle_from_straight_in_rads))
            # delta_d = kd * (self.__prev_delta - delta)
            # delta += delta_d
            # kp = 0.1
            # p += kp * delta_d
            action += np.array([p, delta])
            # self.__prev_delta = delta
            # print(delta)
            self.__env.step(action)

        if key_handler[key.RETURN]:
            im = Image.fromarray(obs)
            im.save("pic.png")

        if done:
            print("done!")
            self.__env.reset()
            self.__env.render()

        self.__env.render()

        if key_handler[key.BACKSPACE]:
            print("RESET")
            prev_delta = 0
            env.reset()
            env.render()

    def main(self):
        pyglet.clock.schedule_interval(self.update, 1.0 / self.__env.unwrapped.frame_rate)
        # Enter main event loop
        pyglet.app.run()
        self.__env.close()


class Graph:
    def __init__(self, path):
        json_graph = json.load(open(path))
        self.__graph = []
        for i in json_graph["graph"]:
            self.__graph.append(i)

    def get_point_id(self, x, y):
        index = None
        x -= 1
        y -= 1
        for i in range(len(self.__graph)):
            if -0.6 < x - self.__graph[i]["cord_x"] < 0.6 and -0.6 < y - self.__graph[i]["cord_y"] < 0.6:
                index = i
        return self.__graph[index]["id"]

    def get_point_cords(self, id):
        index = None
        for i in range(len(self.__graph)):
            if self.__graph[i]["id"] == id:
                index = i
        x = self.__graph[index]["cord_x"]
        y = self.__graph[index]["cord_y"]
        return x, y

    def get_neighbours_id(self, point_id):
        index = None
        for i in range(len(self.__graph)):
            if self.__graph[i]["id"] == point_id:
                index = i
        ids = []
        for i in range(len(self.__graph[index]["neighbors_data"])):
            ids.append(self.__graph[index]["neighbors_data"][i]["id"])
        return ids


class Navi:
    def __init__(self, class_graph):
        self.__graph = class_graph
        self.__way = []

    def generate_way(self, start_point_id, end_point_id):
        id = None
        prev_id = start_point_id
        self.__way = [prev_id]
        while id != end_point_id:
            id = self.__graph.get_neighbours_id(prev_id)[0]
            for i in range(len(self.__way)):
                if self.__way[i] == id:
                    for g in range(len(self.__graph.get_neighbours_id(prev_id))):
                        if self.__graph.get_neighbours_id(prev_id)[g] != self.__way[i]:
                            id = self.__graph.get_neighbours_id(prev_id)[g]
            self.__way.append(id)
            prev_id = id
        return self.__way

    def next_step(self, x, y):
        id = self.__graph.get_point_id(x, y)
        print(id)
        point_index = None
        for i in range(len(self.__way)):
            if self.__way[i] == id:
                point_index = i
                break
        if point_index == len(self.__way) - 1:
            return "arrived"
        else:
            prev_x, prev_y = self.__graph.get_point_cords(self.__way[point_index - 1])
            real_x, real_y = self.__graph.get_point_cords(id)
            next_x, next_y = self.__graph.get_point_cords(self.__way[point_index + 1])
            print(prev_x, prev_y, real_x, real_y, next_x, next_y)
            prev_angle = 0
            if prev_x > real_x:
                prev_angle = -180
            elif prev_x < real_x:
                prev_angle = 0
            elif prev_y > real_y:
                prev_angle = -270
            elif prev_y < real_y:
                prev_angle = -90
            next_angle = 0
            if real_x > next_x:
                next_angle = 180
            elif real_x < next_x:
                next_angle = 0
            elif real_y > next_y:
                next_angle = 270
            elif real_y < next_y:
                next_angle = 90
            angle = prev_angle + next_angle
            print(angle)
            if angle > 180:
                angle = 180 - angle
            elif angle < -180:
                angle = -180 - angle
            if angle == 0:
                return "forward"
            elif angle == 90:
                return "right"
            elif angle == -90:
                return "left"


graph = Graph("duckietown map graph exemple.json")
navi = Navi(graph)
way = navi.generate_way(0, 2)
print(way)
driver = Driver(env, navi)
driver.main()
