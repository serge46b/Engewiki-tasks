from tkinter import *
from tkinter import messagebox
import tkinter as tk
import time

TK = Tk()
app_running = True

size_canvas_x = 600
size_canvas_y = 600

playGround_size_x = 400
playGround_size_y = 400

field_size_x = 8
field_size_y = 8

field_centering_x = 1
field_centering_y = 1

letter_dict = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H"}

x_size = playGround_size_x // field_size_x
y_size = playGround_size_x // field_size_y

bricks = []

ship_dict = {"2": [0, 0, 0, 0, 2, 0], "3": [0, 0, 0, 0, 2, 0], "4": [0, 0, 0, 0, 1, 0]}
selected_ship = 0
ship_placed = 0
ship_rotation = 0

x_ship_prev = 0
y_ship_prev = 0
x_ship_size_prev = 0
y_ship_size_prev = 0
step_prev = 1
event_prev = None


def place_ship(x, y, x_ship_size, y_ship_size):
    global x_ship_prev, y_ship_prev, x_ship_size_prev, y_ship_size_prev, step_prev, ship_placed
    if x != x_ship_prev or y != y_ship_prev or x_ship_size != x_ship_size_prev or y_ship_size != y_ship_size_prev:
        for i in range(x_ship_prev, x_ship_prev + x_ship_size_prev, step_prev):
            for g in range(y_ship_prev, y_ship_prev + y_ship_size_prev, step_prev):
                brick = bricks[g][i]
                brick.b_type = "field"
                brick.draw(canvas)
        #print(x, y, x_ship_size, y_ship_size)
        deny = 0
        step = 1
        if x_ship_size < 0 or y_ship_size < 0:
            step = -1
        for i in range(x, x + x_ship_size, step):
            for g in range(y, y + y_ship_size, step):
                brick = bricks[g][i]
                if brick.b_type == "ship":
                    deny = 1
        if deny == 0:
            for i in range(x, x + x_ship_size, step):
                for g in range(y, y + y_ship_size, step):
                    brick = bricks[g][i]
                    brick.b_type = "ship"
                    brick.draw(canvas)
            x_ship_prev = x
            y_ship_prev = y
            x_ship_size_prev = x_ship_size
            y_ship_size_prev = y_ship_size
            step_prev = step
            ship_placed = 1
        else:
            for i in range(x_ship_prev, x_ship_prev + x_ship_size_prev, step_prev):
                for g in range(y_ship_prev, y_ship_prev + y_ship_size_prev, step_prev):
                    brick = bricks[g][i]
                    brick.b_type = "ship"
                    brick.draw(canvas)


def interface_clicked(event):
    global ship_rotation
    ship_rotation += 90
    if ship_rotation > 270:
        ship_rotation = 0
    clicked(event_prev)


def clicked(event):
    global event_prev
    event_prev = event
    if selected_ship != 0:
        x_brick = event.x // x_size - field_centering_x
        y_brick = event.y // y_size - field_centering_y
        denied_fields_x = 0
        denied_fields_y = 0
        if ship_rotation == 0:
            denied_fields_y = 0 - selected_ship + 1
        elif ship_rotation == 90:
            denied_fields_x = 0 - selected_ship + 1
        elif ship_rotation == 180:
            denied_fields_y = selected_ship - 1
        elif ship_rotation == 270:
            denied_fields_x = selected_ship - 1
        # print(denied_fields_x, denied_fields_y)
        deny = 1
        if 1 <= x_brick <= field_size_x and 1 <= y_brick <= field_size_y:
            if denied_fields_x >= 0 and denied_fields_y >= 0:
                if denied_fields_x < x_brick <= field_size_x and denied_fields_y < y_brick <= field_size_y:
                    # print("ok", x_brick, y_brick)
                    deny = 0
            else:
                if 1 <= x_brick <= field_size_x + denied_fields_x and 1 <= y_brick <= field_size_y + denied_fields_y:
                    # print("ok", x_brick, y_brick)
                    deny = 0
            if deny == 0:
                if ship_rotation >= 180:
                    denied_fields_y += 1
                    denied_fields_x += 1
                else:
                    denied_fields_x -= 1
                    denied_fields_y -= 1
                place_ship(x_brick, y_brick, -1 * denied_fields_x, -1 * denied_fields_y)
                draw_interface_ship(ship_dict["4"][0], ship_dict["4"][1], ship_dict["4"][2], ship_dict["4"][3], False,
                                    canvas)


def ship_selected(event):
    global ship_dict
    global selected_ship
    global x_ship_prev, y_ship_prev, x_ship_size_prev, y_ship_size_prev, step_prev, ship_placed, event_prev
    if ship_dict["4"][0] <= event.x <= ship_dict["4"][2] and ship_dict["4"][1] <= event.y <= ship_dict["4"][3]:
        if selected_ship != 0 and ship_placed == 1:
            ship_dict[str(selected_ship)][4] -= 1
            ship_placed = 0
            x_ship_prev, y_ship_prev, x_ship_size_prev, y_ship_size_prev, event_prev = 0, 0, 0, 0, 0
            step_prev = 1
        if ship_dict["4"][4] > 0:
            selected_ship = 4
            draw_interface_ship(ship_dict["4"][0], ship_dict["4"][1], ship_dict["4"][2], ship_dict["4"][3], True,
                                canvas)
            draw_interface_ship(ship_dict["3"][0], ship_dict["3"][1], ship_dict["3"][2], ship_dict["3"][3], False,
                                canvas)
            draw_interface_ship(ship_dict["2"][0], ship_dict["2"][1], ship_dict["2"][2], ship_dict["2"][3], False,
                                canvas)
        else:
            draw_interface_ship(ship_dict["4"][0], ship_dict["4"][1], ship_dict["4"][2], ship_dict["4"][3], False,
                                canvas)
            selected_ship = 0

    elif ship_dict["3"][0] <= event.x <= ship_dict["3"][2] and ship_dict["3"][1] <= event.y <= ship_dict["3"][3]:
        if selected_ship != 0 and ship_placed == 1:
            ship_dict[str(selected_ship)][4] -= 1
            ship_placed = 0
            x_ship_prev, y_ship_prev, x_ship_size_prev, y_ship_size_prev, event_prev = 0, 0, 0, 0, 0
            step_prev = 1
        if ship_dict["3"][4] > 0:
            selected_ship = 3
            draw_interface_ship(ship_dict["4"][0], ship_dict["4"][1], ship_dict["4"][2], ship_dict["4"][3], False,
                                canvas)
            draw_interface_ship(ship_dict["3"][0], ship_dict["3"][1], ship_dict["3"][2], ship_dict["3"][3], True,
                                canvas)
            draw_interface_ship(ship_dict["2"][0], ship_dict["2"][1], ship_dict["2"][2], ship_dict["2"][3], False,
                                canvas)
        else:
            draw_interface_ship(ship_dict["3"][0], ship_dict["3"][1], ship_dict["3"][2], ship_dict["3"][3], False,
                                canvas)
            selected_ship = 0

    elif ship_dict["2"][0] <= event.x <= ship_dict["2"][2] and ship_dict["2"][1] <= event.y <= ship_dict["2"][3]:
        if selected_ship != 0 and ship_placed == 1:
            ship_dict[str(selected_ship)][4] -= 1
            ship_placed = 0
            x_ship_prev, y_ship_prev, x_ship_size_prev, y_ship_size_prev, event_prev = 0, 0, 0, 0, 0
            step_prev = 1
        if ship_dict["2"][4] > 0:
            selected_ship = 2
            draw_interface_ship(ship_dict["4"][0], ship_dict["4"][1], ship_dict["4"][2], ship_dict["4"][3], False,
                                canvas)
            draw_interface_ship(ship_dict["3"][0], ship_dict["3"][1], ship_dict["3"][2], ship_dict["3"][3], False,
                                canvas)
            draw_interface_ship(ship_dict["2"][0], ship_dict["2"][1], ship_dict["2"][2], ship_dict["2"][3], True,
                                canvas)
        else:
            draw_interface_ship(ship_dict["2"][0], ship_dict["2"][1], ship_dict["2"][2], ship_dict["2"][3], False,
                                canvas)
            selected_ship = 0


def player_shoot(event):
    print("shoot")


class Brick:
    def __init__(self, x, y, x_dimen, y_dimen, b_type):
        self.x = x
        self.y = y
        self.x_dimen = x_dimen
        self.y_dimen = y_dimen
        self.b_type = b_type

    def check_cord(self, x, y):
        return self.x <= x <= self.x + self.x_dimen and self.y <= y <= self.y + self.y_dimen

    def draw(self, cnvs, text=None, text_color="black", fill_color="white", font_size=20):
        if self.b_type == "sign":
            cnvs.create_rectangle(self.x, self.y, self.x + self.x_dimen, self.y + self.y_dimen, fill=fill_color)
            txt = cnvs.create_text(self.x + self.x_dimen // 2, self.y + self.y_dimen // 2,
                                   font="Arial " + str(font_size),
                                   fill=text_color)
            idx = cnvs.index(txt, tk.END)
            cnvs.insert(txt, idx, text)
        elif self.b_type == "interface":
            cnvs.create_rectangle(self.x, self.y, self.x + self.x_dimen, self.y + self.y_dimen, fill=fill_color,
                                  tags="click" + str(self.x))
            txt = cnvs.create_text(self.x + self.x_dimen // 2, self.y + self.y_dimen // 2,
                                   font="Arial " + str(font_size),
                                   fill=text_color, tags="click" + str(self.x))
            cnvs.tag_bind("click" + str(self.x), "<Button-1>", interface_clicked)
            idx = cnvs.index(txt, tk.END)
            cnvs.insert(txt, idx, text)
        elif self.b_type == "after_start_field":
            cnvs.create_rectangle(self.x, self.y, self.x + self.x_dimen, self.y + self.y_dimen, fill="cyan",
                                  outline="white")
        elif self.b_type == "after_start_ship":
            cnvs.create_rectangle(self.x, self.y, self.x + self.x_dimen, self.y + self.y_dimen, fill="red",
                                  outline="white")
        elif self.b_type == "computer_field":
            cnvs.create_rectangle(self.x, self.y, self.x + self.x_dimen, self.y + self.y_dimen, fill="cyan",
                                  outline="white", tags="click" + str(self.x))
            cnvs.tag_bind("click" + str(self.x), "<Button-1>", player_shot)
        else:
            border_color = ""
            if self.b_type == "field":
                fill_color = "cyan"
                border_color = "white"
            elif self.b_type == "ship":
                fill_color = "red"
                border_color = "white"
            cnvs.create_rectangle(self.x, self.y, self.x + self.x_dimen, self.y + self.y_dimen, fill=fill_color,
                                  outline=border_color, tags="click" + str(self.x))
            cnvs.tag_bind("click" + str(self.x), "<B1-Motion>", clicked)


def draw_playground(x_fld_size, y_fld_size, field_ctr_x, field_ctr_y, cnvs, aditionals=""):
    global x_size
    global y_size
    global bricks
    for i in range(field_ctr_y, y_fld_size + field_ctr_y + 1):
        mas = []
        for g in range(field_ctr_x, x_fld_size + field_ctr_x + 1):
            if i == field_ctr_y:
                brick = Brick(x_size * g, y_size * i, x_size, y_size, "sign")
                text = "..."
                if g - field_ctr_x != 0:
                    text = g - field_ctr_x
                brick.draw(cnvs, text)
            elif g == field_ctr_x:
                brick = Brick(x_size * g, y_size * i, x_size, y_size, "sign")
                brick.draw(cnvs, letter_dict[i - field_ctr_y])
            else:
                brick = Brick(x_size * g, y_size * i, x_size, y_size, aditionals + "field")
                brick.draw(cnvs)
            mas.append(brick)
        bricks.append(mas)


def draw_interface_ship(x, y, x2, y2, is_selected, cnvs):
    if is_selected:
        border_color = "green"
    else:
        border_color = "white"
    cnvs.create_rectangle(x, y, x2, y2, fill="red", outline=border_color, tags="click" + str(x))
    txt = cnvs.create_text((x + x2) // 2, (y + y2) // 2, font="Arial 12", fill="white", tags="click" + str(x))
    idx = cnvs.index(txt, tk.END)
    cnvs.insert(txt, idx, (y2 - y) // 20)
    lbl = ship_dict[str((y2 - y) // 20)][5]
    color = "black"
    if ship_dict[str((y2 - y) // 20)][4] == 0:
        color = "gray"
    cnvs.itemconfig(lbl, fill=color, text=str(ship_dict[str((y2 - y) // 20)][4]) + "x")
    cnvs.tag_bind("click" + str(x), "<Button-1>", ship_selected)


def draw_interface(x_cnvs_size, y_cnvs_size, cnvs):
    x_cnvs_size -= 15
    y_cnvs_size -= 15
    global ship_dict

    x = x_cnvs_size - 10
    y = y_cnvs_size - 80

    lbl = cnvs.create_text((x + x_cnvs_size) // 2, y - 10, font="Arial 12", fill="black")
    idx = cnvs.index(lbl, tk.END)
    cnvs.insert(lbl, idx, str(ship_dict["4"][4]) + "x")

    ship_dict["4"][0] = x
    ship_dict["4"][1] = y
    ship_dict["4"][2] = x_cnvs_size
    ship_dict["4"][3] = y_cnvs_size
    ship_dict["4"][5] = lbl

    draw_interface_ship(x, y, x_cnvs_size, y_cnvs_size, False, cnvs)

    x_cnvs_size -= 15
    x = x_cnvs_size - 10
    y = y_cnvs_size - 60

    lbl = cnvs.create_text((x + x_cnvs_size) // 2, y - 10, font="Arial 12", fill="black")
    idx = cnvs.index(lbl, tk.END)
    cnvs.insert(lbl, idx, str(ship_dict["3"][4]) + "x")

    ship_dict["3"][0] = x
    ship_dict["3"][1] = y
    ship_dict["3"][2] = x_cnvs_size
    ship_dict["3"][3] = y_cnvs_size
    ship_dict["3"][5] = lbl

    draw_interface_ship(x, y, x_cnvs_size, y_cnvs_size, False, cnvs)

    x_cnvs_size -= 15
    x = x_cnvs_size - 10
    y = y_cnvs_size - 40

    lbl = cnvs.create_text((x + x_cnvs_size) // 2, y - 10, font="Arial 12", fill="black")
    idx = cnvs.index(lbl, tk.END)
    cnvs.insert(lbl, idx, str(ship_dict["2"][4]) + "x")

    ship_dict["2"][0] = x
    ship_dict["2"][1] = y
    ship_dict["2"][2] = x_cnvs_size
    ship_dict["2"][3] = y_cnvs_size
    ship_dict["2"][5] = lbl

    draw_interface_ship(x, y, x_cnvs_size, y_cnvs_size, False, cnvs)

    y_cnvs_size -= 80
    x = x_cnvs_size - 25
    y = y_cnvs_size - 80

    rotate_btn = Brick(x, y, 50, 50, "interface")
    rotate_btn.draw(cnvs, text="rotate", font_size=11)


def on_closing():
    global app_running
    if messagebox.askokcancel("Выход из игры", "Хотите выйти из игры?"):
        app_running = False
        TK.destroy()


start_flag = 0
my_field = []


def start_game(event):
    global start_flag, my_field, bricks
    for i in range(1, len(bricks)):
        mas = []
        for g in range(1, len(bricks[i])):
            mas.append(bricks[i][g].b_type)
        my_field.append(mas)
    canvas.delete("all")
    canvas.create_rectangle(0, 0, size_canvas_x, size_canvas_y, fill="white")
    bricks = []
    for i in range(field_centering_y, field_size_y + field_centering_y + 1):
        mas = []
        for g in range(field_centering_x, field_size_x + field_centering_x + 1):
            if i == field_centering_y:
                brick = Brick(x_size * g, y_size * i, x_size, y_size, "sign")
                text = "..."
                if g - field_centering_x != 0:
                    text = g - field_centering_x
                    brick.draw(canvas, text)
            elif g == field_centering_x:
                brick = Brick(x_size * g, y_size * i, x_size, y_size, "sign")
                brick.draw(canvas, letter_dict[i - field_centering_y])
            else:
                brick = Brick(x_size * g, y_size * i, x_size, y_size,
                              "after_start_" + my_field[i - field_centering_y - 1][g - field_centering_x - 1])
                brick.draw(canvas)
            mas.append(brick)
        bricks.append(mas)
    start_flag = 1


TK.protocol("WM_DELETE_WINDOW", on_closing)
TK.title("Ваше поле")
TK.resizable(0, 0)
TK.wm_attributes("-topmost", 1)
canvas = Canvas(TK, width=size_canvas_x, height=size_canvas_y, bd=0, highlightthickness=0)
canvas.create_rectangle(0, 0, size_canvas_x, size_canvas_y, fill="white")
draw_playground(field_size_x, field_size_y, field_centering_x, field_centering_y, canvas)
draw_interface(size_canvas_x, size_canvas_y, canvas)
canvas.pack()
TK.update()

while ship_dict["4"][4] + ship_dict["3"][4] + ship_dict["2"][4] > 1 or ship_placed == 0:
    if app_running:
        TK.update_idletasks()
        TK.update()
    time.sleep(0.005)

canvas.create_rectangle(80, size_canvas_y - 60, size_canvas_x - 80, size_canvas_y - 20, fill="white",
                        outline="black", tags="start")
start_text = canvas.create_text((80 + size_canvas_x - 80) // 2, size_canvas_y - 40, font="Arial 12", fill="black",
                                tags="start")
idx = canvas.index(start_text, tk.END)
canvas.insert(start_text, idx, "Start!")
canvas.tag_bind("start", "<Button-1>", start_game)

while start_flag == 0:
    if app_running:
        TK.update_idletasks()
        TK.update()
    time.sleep(0.005)

size_canvas2_x = 450
size_canvas2_y = 450

field2_centering2_x = 0
field2_centering2_y = 0

TK2 = Tk()
TK2.protocol("WM_DELETE_WINDOW", on_closing)
TK2.title("Поле противника")
TK2.resizable(0, 0)
TK2.wm_attributes("-topmost", 1)
canvas2 = Canvas(TK2, width=size_canvas2_x, height=size_canvas2_y, bd=0, highlightthickness=0)
canvas2.create_rectangle(0, 0, size_canvas_x, size_canvas_y, fill="white")
draw_playground(field_size_x, field_size_y, field2_centering2_x, field2_centering2_y, canvas2, aditionals="computer_")
canvas2.pack()
TK2.update()

while app_running:
    if app_running:
        TK.update_idletasks()
        TK.update()
    time.sleep(0.005)
