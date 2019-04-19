from tkinter import Image, Canvas, Tk, Button, YES, BOTH, LEFT, RIGHT, BOTTOM
import PIL
from PIL import Image, ImageDraw
import pickle


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


def save():
    # global image_number
    img_pkl_number = pickle.load(open("save.p", "rb"))
    filename = f'sad_{img_pkl_number}.bmp'
    # save the file
    image1.save(filename)
    # resize the image based on the width (REQUIRED TO BE SQUARE)
    basewidth = 28
    img = Image.open(filename)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save(filename)
    # open the png image to transform into bmp
    # image2 = Image.open(filename)
    # image2.save('welp.bmp')
    # image_number += 1
    img_pkl_number += 1
    pickle.dump(img_pkl_number, open("save.p", "wb"))


def restart():
    img_pkl_number = 0
    pickle.dump(img_pkl_number, open("save.p", "wb"))


def clean():
    cv.delete("all")
    draw.rectangle((0, 0, 560, 560), "white")

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
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="save", command=save)
btn_save.pack(side=LEFT)

btn_clean = Button(text="clean", command=clean)
btn_clean.pack(side=LEFT)

btn_process = Button(text="compute", command=save)
btn_process.pack(side=RIGHT)

btn_restart = Button(text="restart num", command=restart)
btn_restart.pack(side=BOTTOM)

master.mainloop()
