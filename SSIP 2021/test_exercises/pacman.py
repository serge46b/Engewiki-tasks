from time import sleep
from random import choice

scores = 0
game_over = False
obj_lst = []
on_start_layout = [['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
                   ['w', 'g', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'w'],
                   ['w', 'e', 'w', 'w', 'w', 'e', 'w', 'w', 'e', 'w'],
                   ['w', 'e', 'e', 'e', 'e', 'e', 'e', 'w', 'e', 'w'],
                   ['w', 'w', 'w', 'e', 'w', 'w', 'e', 'w', 'e', 'w'],
                   ['w', 'e', 'e', 'e', 'p', 'e', 'e', 'w', 'e', 'w'],
                   ['w', 'e', 'w', 'w', 'w', 'e', 'w', 'w', 'e', 'w'],
                   ['w', 'e', 'w', 'w', 'e', 'e', 'e', 'w', 'e', 'w'],
                   ['w', 'e', 'e', 'e', 'e', 'w', 'e', 'e', 'g', 'w'],
                   ['w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']]


class Scene:
    def __init__(self, refresh_rate):
        self.ref_rate = refresh_rate

    def generate_map(self, layout):
        global obj_lst
        for y in range(len(layout)):
            row_objs = []
            for x in range(len(layout[y])):
                if layout[y][x] == 'w':
                    row_objs.append(Wall(x=x, y=y))
                elif layout[y][x] == 'e':
                    row_objs.append(Eat(x=x, y=y))
                elif layout[y][x] == 'g':
                    row_objs.append(Ghost(x=x, y=y))
                elif layout[y][x] == 'p':
                    row_objs.append(Pacman(x=x, y=y))
            obj_lst.append(row_objs)
        self.refresh_map()

    def refresh_map(self):
        cords = []
        global obj_lst
        for i in range(len(obj_lst)):
            row_cords = []
            for j in range(len(obj_lst[i])):
                row_cords.append([obj_lst[i][j].get_cord(), type(obj_lst[i][j]).__name__, obj_lst[i][j]])
            for g in range(0, len(row_cords) - 1):
                for h in range(g + 1, len(row_cords)):
                    if row_cords[g][0][0] > row_cords[h][0][0]:
                        row_cords[g], row_cords[h] = row_cords[h], row_cords[g]
            cords.append(row_cords)
        for i in range(len(cords)):
            for g in range(0, len(cords[i]) - 1):
                for h in range(g + 1, len(cords[i])):
                    if cords[g][i][0][1] > cords[h][i][0][1]:
                        cords[g][i], cords[h][i] = cords[h][i], cords[g][i]

        print(cords)
        for y in range(len(cords)):
            prnt_str = ''
            for x in range(len(cords[y])):
                if cords[y][x][1] == 'Wall':
                    prnt_str += '# '
                elif cords[y][x][1] == 'Pacman':
                    prnt_str += '^ '
                    #print(x, y)
                elif cords[y][x][1] == 'Eat':
                    prnt_str += '* '
                elif cords[y][x][1] == 'Ghost':
                    prnt_str += '@ '
                elif cords[y][x][1]:
                    prnt_str += '  '
                else:
                    print('else', cords[1])
                #if x == cords[y][x][0][0] and y == cords[y][x][0][1]:
                #    print("cord ok")
            print(prnt_str)
        print('\n')



class Sprites:
    def __init__(self, x=0, y=0):
        self.sprites = []
        self.x = x
        self.y = y

    """def get_sprites(self):
        clss = Sprites.__subclasses__()
        retlst = []
        for i in clss:
            sclss = i.__subclasses__()
            if not sclss:
                retlst.append(i)
            else:
                for s in sclss:
                    retlst.append(s)
        return retlst"""

    def get_cord(self):
        return [self.x, self.y]


class Field(Sprites):
    pass


class Eat(Sprites):
    def be_eaten(self):
        global scores, obj_lst
        scores += 1
        obj_lst[self.y][self.x] = Field(x=self.x, y=self.y)
        self.__del__()

    def __del__(self):
        pass


class Cherry(Sprites):
    def be_eaten(self):
        global scores, obj_lst
        scores += 5
        #obj_lst[self.y][self.x] = Field(x=self.x, y=self.y)
        self.__del__()

    def __del__(self):
        pass


class Wall(Sprites):
    pass


class Creature(Sprites):

    def move_to(self, step):
        mx = 0
        my = 0
        if step == "up":
            my = -1
        elif step == "down":
            my = 1
        elif step == "right":
            mx = 1
        elif step == "left":
            mx = -1
        else:
            raise RuntimeError("unknown step direction")
        cords = []
        for i in range(len(obj_lst)):
            for j in obj_lst[i]:
                if j != self:
                    cords.append([j.get_cord(), type(j).__name__, j])
        for i in cords:
            if self.x + mx == i[0][0] and self.y + my == i[0][1]:
                if i[1] == "Wall":
                    mx, my = 0, 0
                    print('wall')
                elif i[1] == "Eat" or i[1] == "Cherry" and type(self).__name__ != 'Ghost':
                    i[2].be_eaten()
                    print('eat')
                elif i[1] == "Ghost" and type(self).__name__ != 'Ghost':
                    global game_over
                    game_over = True
                    print('ghost')
        self.x += mx
        self.y += my


class Pacman(Creature):
    pass


class Ghost(Creature):
    pass


sprts = Sprites()
scene = Scene(1)
scene.generate_map(on_start_layout)
refresh_time = 1 / scene.ref_rate

while not game_over:
    #choice(['up', 'right', 'down', 'left'])
    for y in range(len(obj_lst)):
        for x in range(len(obj_lst[y])):
            if type(obj_lst[y][x]).__name__ == 'Pacman' or type(obj_lst[y][x]).__name__ == 'Ghost':
                obj_lst[y][x].move_to(choice(['up', 'right', 'down', 'left']))
                print(obj_lst[y][x].get_cord(), type(obj_lst[y][x]).__name__)

    print(obj_lst)
    scene.refresh_map()
    sleep(refresh_time)
if game_over:
    print("game over")
