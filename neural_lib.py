import numpy 
import matplotlib.pyplot as pyplot
import PIL.Image as Image
import pickle

from os import listdir
from os.path import isfile, join


def softmax(A):
    expA = numpy.exp(A)
    expA = numpy.nan_to_num(expA)
    # print(expA)
    return expA / expA.sum()


def sigmoid(x):
    x = numpy.nan_to_num(x)
    return 1/(1+numpy.exp(-x))


def sigmoid_der(x):
    x = numpy.nan_to_num(x)
    return sigmoid(x) * (1-sigmoid(x))


def feedforward(feature_set, wh, bh, wo, bo):
    # Phase 1
    zh = numpy.dot(feature_set, wh) + bh
    ah = sigmoid(zh)

    # Phase 2
    zo = numpy.dot(ah, wo) + bo
    ao = softmax(zo)

    return (ao, ah, zo, zh)


def backpropagation(feature_set, one_hot_labels, ao, ah, wo, zh):
    # Phase 1
    dcost_dzo = ao - one_hot_labels
    dzo_dwo = ah

    dcost_wo = numpy.dot(dzo_dwo.T, dcost_dzo)

    dcost_bo = dcost_dzo

    # Phase 2
    dzo_dah = wo
    dcost_dah = numpy.dot(dcost_dzo, dzo_dah.T)
    dah_dzh = sigmoid_der(zh)
    dzh_dwh = feature_set

    dcost_wh = numpy.dot(dzh_dwh.T, dah_dzh * dcost_dah)

    dcost_bh = dcost_dah * dah_dzh

    return (dcost_wo, dcost_bo, dcost_wh, dcost_bh)


def weight_update(wh, dcost_wh, bh, dcost_bh, wo, dcost_wo, bo, dcost_bo, lr):
    wh -= lr * dcost_wh
    bh -= lr * dcost_bh.sum(axis=0)

    wo -= lr * dcost_wo
    bo -= lr * dcost_bo.sum(axis=0)

    return (wh, bh, wo, bo)


def images_to_pickle(path, s_name):
    onlyfiles = [fil for fil in listdir(path) if isfile(join(path, fil))]
    # iterate said array so we get all files into Black white arrays
    print('found: ' + str(len(onlyfiles)) + ' files')
    # new array of images
    images_vector = []
    for fil in onlyfiles:
        # open a image
        filename = fil
        img = Image.open(path+'\\'+filename)
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
    pickle.dump(images_vector, open(s_name, "wb"))
