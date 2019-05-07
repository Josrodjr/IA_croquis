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
        # using the grays
        
        # < -------------- 0 or 255 IMPORTANT ---------------- >

        # black_white[black_white < 128] = 0
        # black_white[black_white >= 128] = 255

        # make sure that the image is exactly the 28*28 size (784)
        # print(black_white.size)
        # concatenate every column of the image into a single array
        single_array = numpy.concatenate(black_white, axis=None)
        # add the image to the images array
        images_vector.append(single_array)
    # dump the compelte data for acess later
    pickle.dump(images_vector, open(s_name, "wb"))

# Another try for a neuralnet forward back gradientdescent


def starter_bias(H_size, O_size):
    Hidden_bias = numpy.full((1 ,H_size), 0.1)
    Output_bias = numpy.full((1 ,O_size), 0.1)
    return Hidden_bias, Output_bias


# funcion de activacion ReLU
def ReLU(vector_tbw):
    return numpy.maximum(0, vector_tbw)


# derivada de funcion de activacion ReLU
def derivada_ReLU(vector_tbw):
    vector_tbw[vector_tbw < 0] = 0
    vector_tbw[vector_tbw > 1] = 1
    return vector_tbw


def costo(predicciones, output_layer):
    costo = numpy.sum((predicciones -  output_layer) ** 2 / y.size)
    return costo


def derivda_costo(predicciones, output_layer):
    return predicciones - output_layer


def feed_forward2(input_matrix, weight_HL, weight_OL, bias_HL, bias_OL):
    # input con peso hidden layer = IHL
    # activacion de el hidden layer = HLA

    # predicciones de output layer
    # output con peso = OCP

    # hidden layer
    IHL = numpy.dot(input_matrix, weight_HL) + bias_HL
    # cambiar fucnion de activacion si truena
    HLA = ReLU(IHL)

    # output layer
    OCP = numpy.dot(HLA, weight_OL) + bias_OL
    # func act cambiar 
    predicciones = ReLU(OCP)

    return (IHL, HLA, OCP, predicciones)


def descenso_gradiente(m, b, X, Y, rate):
    deriv_m = 0
    deriv_b = 0
    N = len(X)
    for i in range(N):
        deriv_m += -2*X[i] * (Y[i] - (m*X[i] + b))
        deriv_b += -2*(Y[i] - (m*X[i] + b))
    
    # restar la derivada en la direccion del decenso
    m -= (deriv_m / float(N)) * rate
    b -= (deriv_b / float(N)) * rate

    return m, b


def backpropagation2(input_matrix, output_layer, weight_HL, weight_OL, rate, HL, OL, outputs):
    # utilizar predicciones
    # IHL, HLA, OCP, predicciones = feed_forward2(input_matrix, weight_HL, weight_OL, bias_HL, bias_OL)

    # Error de la layer de la output layer
    error_OL = (outputs - output_layer) * derivada_ReLU(OL)
    # Error de la hidden layer
    # error_HL = numpy.dot(error_OL, weight_OL) * derivada_ReLU(HL)
    error_HL = numpy.dot(numpy.dot(weight_HL, error_OL.T), derivada_ReLU(HL))

    # Derivadas de pesos para weights
    cost_derivOL = numpy.dot(error_OL.T, ReLU(OL))
    cost_derivHL = numpy.dot(error_HL, weight_OL)

    # Actualizar weights

    weight_OL -= rate * cost_derivOL
    weight_HL -= rate * cost_derivHL

    return (weight_OL, weight_HL)
