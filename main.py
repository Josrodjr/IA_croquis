from tkinter import Image, Canvas, Tk, Button, YES, BOTH, LEFT, RIGHT, BOTTOM
import PIL
from PIL import Image, ImageDraw
import pickle
import numpy 
from neural_lib import feed_forward2

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=1)
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='black', width=35)
    lastx, lasty = x, y


def activate_circle(e):
    global lastx, lasty
    cv.bind('<ButtonRelease-3>', circle)
    lastx, lasty = e.x, e.y


def circle(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_oval((lastx, lasty, x, y), width=1, fill="black")
    #  --- PIL
    draw.ellipse((lastx, lasty, x, y), fill='black', width=35)
    # draw.line((lastx, lasty, x, y), fill='black', width=35)
    lastx, lasty = x, y


def save():
    # global image_number
    img_pkl_number = pickle.load(open("save.p", "rb"))
    filename = f'img_{img_pkl_number}.bmp'
    # save the file
    image1.save(filename)
    # resize the image based on the width (REQUIRED TO BE SQUARE)
    basewidth = 28
    img = Image.open(filename)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save(filename)
    img_pkl_number += 1
    pickle.dump(img_pkl_number, open("save.p", "wb"))
    # LINE ONLY TO KEEP IMAGES AT 0
    restart()


def restart():
    img_pkl_number = 0
    pickle.dump(img_pkl_number, open("save.p", "wb"))


def clean():
    cv.delete("all")
    draw.rectangle((0, 0, 560, 560), "white")


def compute():
    # load the "trained" NN weights and biases
    weight_HL = 0
    weight_OL = 0
    bias_HL = 0
    bias_OL = 0
    weight_HL, weight_OL, bias_HL, bias_OL = pickle.load(open("weights_bias.p", "rb"))
    # load the image into a matrix
    image_vector = []
    img = Image.open(r'C:\Users\Josro\Documents\GitHub\IA_croquis'+'\\'+'img_0.bmp')
    gray = img.convert('L')
    black_white = numpy.asarray(gray).copy()
    single_array = numpy.concatenate(black_white, axis=None)
    image_vector.append(single_array)
    image_matrix = numpy.asarray(image_vector)
    IHL, HLA, OCP, predicciones = feed_forward2(image_matrix, weight_HL, weight_OL, bias_HL, bias_OL)
    print(predicciones)

master = Tk()
master.title("Paint a la tortix")

lastx, lasty = None, None
image_number = 0

cv = Canvas(master, width=560, height=560, bg='white')
# --- PIL
# black background on BMP or transparent on PNG
# image1 = PIL.Image.new('RGBA', (560, 560), (0, 0, 0, 0))
# white backgorund
image1 = PIL.Image.new('RGB', (560, 560), 'white')
draw = ImageDraw.Draw(image1)

cv.bind('<1>', activate_paint)
cv.bind('<3>', activate_circle)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack(side=LEFT)

btn_clean = Button(text="clean", command=clean)
btn_clean.pack(side=LEFT)

btn_process = Button(text="compute", command=compute)
btn_process.pack(side=RIGHT)

btn_restart = Button(text="restart num", command=restart)
btn_restart.pack(side=BOTTOM)

master.mainloop()
