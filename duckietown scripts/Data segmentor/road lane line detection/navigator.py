import json


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
        #self.__way = []
    """
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
    """
    """
    def generate_way(self, start_point_id, end_point_id):

        return
    """
    def get_direction_from_point(self, now_id, next_id, now_angle):
        real_x, real_y = self.__graph.get_point_cords(now_id)
        next_x, next_y = self.__graph.get_point_cords(next_id)
        print(real_x, real_y, next_x, next_y)
        prev_angle = 0
        if -45 < now_angle < 45:
            prev_angle = -180
        elif -180 < now_angle < -135 or 180 > now_angle > 135:
            prev_angle = 0
        elif 135 > now_angle > 45:
            prev_angle = -270
        elif -45 > now_angle > -135:
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
        print(prev_angle, next_angle, angle)
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
        else:
            return "error"
    """
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
            """
