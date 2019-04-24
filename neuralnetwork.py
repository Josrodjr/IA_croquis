import numpy 
import matplotlib.pyplot as pyplot
import pickle

# local library for the neuralnetwork methods
from neural_lib import images_to_pickle, feedforward, backpropagation, weight_update, sigmoid

# set a randomseed
numpy.random.seed(69)

# numpy.seterr(divide='ignore', invalid='ignore')
# numpy.warnings.filterwarnings('ignore')

# ******************************* HUEVO ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Documents\GitHub\IA_croquis\training\huevo"
savename = f"huevo.p"

images_to_pickle(mypath, savename)
images_vector_huevo = pickle.load(open(savename, "rb"))

# ******************************* MICKEY ***********************************
# get all files in a path
mypath = r"C:\Users\Josro\Documents\GitHub\IA_croquis\training\mickey"
savename = f"mickey.p"

images_to_pickle(mypath, savename)
images_vector_mickey = pickle.load(open(savename, "rb"))

# ******************************* FELIZ ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Documents\GitHub\IA_croquis\training\feliz"
savename = f"feliz.p"

images_to_pickle(mypath, savename)
images_vector_feliz = pickle.load(open(savename, "rb"))

# ***************************** ACTUAL NN **********************************

# actual 0 = huevo, 1 = mickey, 2 = feliz
labels_for_data = numpy.array([0]*251 + [1]*251 + [2]*251)

ones_labels = numpy.zeros((753, 3))

for i in range(len(ones_labels)):
    ones_labels[i, labels_for_data[i]] = 1

feature_set = numpy.vstack([images_vector_huevo, images_vector_mickey, images_vector_feliz])

# get the number of rows
instances = feature_set.shape[0]
# get the number of columns
attributes = feature_set.shape[1]
# neural network size
hidden_nodes = 784
output_labels = 3

wh = numpy.random.rand(attributes, hidden_nodes)
bh = numpy.random.randn(hidden_nodes)

wo = numpy.random.rand(hidden_nodes, output_labels)
bo = numpy.random.randn(output_labels)
lr = 0.001

error_cost = []

for epoch in range(5000):
    # do feedforward
    ao, ah, zo, zh = feedforward(feature_set, wh, bh, wo, bo)
    # do backpropagation
    dcost_wo, dcost_bo, dcost_wh, dcost_bh = backpropagation(feature_set, ones_labels, ao, ah, wo, zh)
    # update the weighs
    wh, bh, wo, bo = weight_update(wh, dcost_wh, bh, dcost_bh, wo, dcost_wo, bo, dcost_bo, lr)

    if epoch % 200 == 0:
        ao = numpy.nan_to_num(ao)
        loss = numpy.sum(-ones_labels * numpy.log(ao))
        print('Loss function value: ', loss)
        error_cost.append(loss)

# test this zhisnit
result = sigmoid(numpy.dot(images_vector_mickey[150], wo) + bo)
print(result)

# def feedforward(feature_set, wh, bh, wo, bo):          return (ao, ah, zo, zh)
# def backpropagation(feature_set, one_hot_labels, ao, ah, wo, zh):             return (dcost_wo, dcost_bo, dcost_wh, dcost_bh)
# def weight_update(wh, dcost_wh, bh, dcost_bh, wo, dcost_wo, bo, dcost_bo, lr):           return (wh, bh, wo, bo)