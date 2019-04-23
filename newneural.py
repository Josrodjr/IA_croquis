import numpy 
import matplotlib.pyplot as pyplot
import PIL.Image as Image
import pickle

from os import listdir
from os.path import isfile, join

numpy.random.seed(69)

# get the images we have stored into a csv of data

# try to open the directory of te file we are workig with

# # get the files that are on this path into an array of files
# onlyfiles = [file for file in listdir(mypath) if isfile(join(mypath, file))]
# # iterate said array so we get all files into Black white arrays
# for file in onlyfiles:
#     # open a image
#     filename = file
#     img = Image.open(mypath+'\\'+filename)
#     # get the image into an array of bytes
#     # convert the image to pure Blackwhite (Not RGB)
#     gray = img.convert('L')
#     # Numpy for comverting the pixels into pure white (255) and black (0)
#     black_white = numpy.asarray(gray).copy()
#     black_white[black_white < 128] = 0
#     black_white[black_white >= 128] = 255
#     # img_bytes = numpy.array(img)
#     # make sure that the image is exactly the 28*28 size (784)
#     print(black_white.size)


def images_to_pickle(path, s_name):
    onlyfiles = [file for file in listdir(mypath) if isfile(join(mypath, file))]
    # iterate said array so we get all files into Black white arrays
    print('found: ' + str(len(onlyfiles)) + ' files')
    # new array of images
    images_vector = []
    for file in onlyfiles:
        # open a image
        filename = file
        img = Image.open(mypath+'\\'+filename)
        # get the image into an array of bytes
        # convert the image to pure Blackwhite (Not RGB)
        gray = img.convert('L')
        # Numpy for comverting the pixels into pure white (255) and black (0)
        black_white = numpy.asarray(gray).copy()
        black_white[black_white < 128] = 0
        black_white[black_white >= 128] = 255
        # make sure that the image is exactly the 28*28 size (784)
        # print(black_white.size)
        # concatenate every column of the image into a single array
        single_array = numpy.concatenate(black_white, axis=None)
        # add the image to the images array
        images_vector.append(single_array)
    # dump the compelte data for acess later
    pickle.dump(images_vector, open(savename, "wb"))

mypath = r"C:\Users\Josro\Documents\GitHub\IA_croquis\training\huevo"
savename = f"huevo.p"

images_to_pickle(mypath, savename)
images_vector_huevo = pickle.load(open(savename, "rb"))

print(len(images_vector_huevo))
