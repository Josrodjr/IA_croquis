import numpy 
import matplotlib.pyplot as pyplot
import pickle
import PIL
import PIL.Image as Image

from os import listdir
from os.path import isfile, join

# set a randomseed
numpy.random.seed(69)


def images_to_bmp(path, newpath):
    onlyfiles = [fil for fil in listdir(path) if isfile(join(path, fil))]
    # iterate said array so we get all files into Black white arrays
    print('found: ' + str(len(onlyfiles)) + ' files')
    imagenumber = 0
    for fil in onlyfiles:
        # open a image
        filename = fil
        img = Image.open(path+'\\'+filename)
        # resize the image based on the width (REQUIRED TO BE SQUARE)
        basewidth = 28
        # img = Image.open(filename)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        # img.save(filename)
        img.save(newpath+'\\'+f'{imagenumber}.bmp', 'BMP')
        imagenumber += 1


testpath = r"C:\Users\Josro\Desktop\training_all\Tree"
newpath = r"C:\Users\Josro\Desktop\training_all\Tree_new"
images_to_bmp(testpath, newpath)