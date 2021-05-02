scores = 0
game_over = False
obj_lst = []


class Scene:
    def __init__(self, sprite_cls, refresh_rate):
        self.sprite_cls = sprite_cls
        self.ref_rate = refresh_rate

    def generate_map(self, layout):
        pass

    def refresh_map(self):
        pass


class Sprites:
    def __init__(self, x=0, y=0, init=False):
        self.sprites = []
        self.x = x
        self.y = y
        if init == False:
            global obj_lst
            obj_lst.append(self)

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


class Eat(Sprites):
    def be_eaten(self):
        global scores, obj_lst
        scores += 1
        obj_lst.remove(self)

    def __del__(self):
        pass


class Cherry(Sprites):
    def be_eaten(self):
        global scores, obj_lst
        scores += 5
        obj_lst.remove(self)

    def __del__(self):
        pass


class Wall(Sprites):
    pass


class Creature(Sprites):

    def move_to(self, step):
        mx = 0
        my= 0
        if step == "up":
            my = 1
        elif step == "down":
            my -= 1
        elif step == "right":
            mx += 1
        elif step == "left":
            mx -= 1
        else:
            raise RuntimeError("unknown step direction")
        cords = []
        for i in obj_lst:
            if i != self:
                cords.append([i.get_cord(), type(i).__name__, i])
        for i in cords:
            if self.x + mx == i[0][0] and self.y + my == i[0][1]:
                if i[1] == "Wall":
                    mx, my = 0, 0
                elif i[1] == "Eat" or i[1] == "Cherry":
                    i[2].be_eaten()
                    self.x += mx
                    self.y += my
                elif i[1] == "Ghost":
                    global game_over
                    game_over = True
                else:
                    self.x += mx
                    self.y += my
            else:
                self.x += mx
                self.y += my


class Pacman(Creature):
    pass


class Ghost(Creature):
    pass

sprts = Sprites(init=True)
eat = Eat(x=1, y=2)
eat2 = Eat(x=0, y=1)
wall = Wall(x=1, y=0)
pacman = Pacman()
ghost = Ghost(x=5, y=5)
print(pacman.x, pacman.y, game_over)
pacman.move_to("right")
print(pacman.x, pacman.y, game_over)
#scene = Scene(sprts, 1)
