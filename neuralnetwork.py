import numpy 
import matplotlib.pyplot as pyplot
import pickle

# local library for the neuralnetwork methods
from neural_lib import images_to_pickle, feed_forward2, backpropagation2, feed_forward3

# set a randomseed
numpy.random.seed(69)

# numpy.seterr(divide='ignore', invalid='ignore')
# numpy.warnings.filterwarnings('ignore')

# ******************************* CIRCULO ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Circle_new"
savename = f"circulo.p"

# images_to_pickle(mypath, savename)
images_vector_circulo = pickle.load(open(savename, "rb"))

# ******************************* CUADRADO ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Square_new"
savename = f"cuadrado.p"

# images_to_pickle(mypath, savename)
images_vector_cuadrado = pickle.load(open(savename, "rb"))

# ******************************* HUEVO ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Egg_new"
savename = f"huevo.p"

# images_to_pickle(mypath, savename)
images_vector_huevo = pickle.load(open(savename, "rb"))

# ******************************* ARBOL ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Tree_new"
savename = f"arbol.p"

# images_to_pickle(mypath, savename)
images_vector_arbol = pickle.load(open(savename, "rb"))

# ******************************* CASA ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\House_new"
savename = f"casa.p"

# images_to_pickle(mypath, savename)
images_vector_casa = pickle.load(open(savename, "rb"))

# ******************************* FELIZ ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Smile_new"
savename = f"feliz.p"

# images_to_pickle(mypath, savename)
images_vector_feliz = pickle.load(open(savename, "rb"))

# ******************************* TRISTE ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Sad_new"
savename = f"triste.p"

# images_to_pickle(mypath, savename)
images_vector_triste = pickle.load(open(savename, "rb"))

# ******************************* INTERR ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Question_new"
savename = f"qmark.p"

# images_to_pickle(mypath, savename)
images_vector_qmark = pickle.load(open(savename, "rb"))

# ******************************* MICKEY ***********************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Mickey_new"
savename = f"mickey.p"

# images_to_pickle(mypath, savename)
images_vector_mickey = pickle.load(open(savename, "rb"))


# ***************************** ACTUAL NN **********************************

# welp
NUMBER_OF_ITERATIONS = 500
# 28*28 por los bmp
INPUT_LAYER_SIZE = 784
HIDDEN_LAYER_SIZE = 9
OUTPUT_LAYER_SIZE = 9

HL = numpy.full((HIDDEN_LAYER_SIZE), numpy.random.rand(1, 1))
OL = numpy.full((OUTPUT_LAYER_SIZE), numpy.random.rand(1, 1))
weight_HL = numpy.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
weight_OL = numpy.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE)
bias_HL = numpy.full((1, HIDDEN_LAYER_SIZE), 0.1)
bias_OL = numpy.full((1, OUTPUT_LAYER_SIZE), 0.1)

# x = numpy.full((INPUT_LAYER_SIZE), 0.1)

# print(numpy.dot(x, weight_HL))
# print(OL)
# print(weight_HL)

# print(len(images_vector_arbol))
# print(images_vector_arbol[0].size)

# create a new matrix for all the data we colected (and all the vectors containing them)
matrix_y_size = len(images_vector_arbol) \
    + len(images_vector_casa) \
    + len(images_vector_circulo) \
    + len(images_vector_cuadrado) \
    + len(images_vector_feliz) \
    + len(images_vector_huevo) \
    + len(images_vector_mickey) \
    + len(images_vector_qmark) \
    + len(images_vector_triste)

ar = numpy.asarray(images_vector_arbol)
ca = numpy.asarray(images_vector_casa)
ci = numpy.asarray(images_vector_circulo)
cu = numpy.asarray(images_vector_cuadrado)
fe = numpy.asarray(images_vector_feliz)
hu = numpy.asarray(images_vector_huevo)
mi = numpy.asarray(images_vector_mickey)
qm = numpy.asarray(images_vector_qmark)
tr = numpy.asarray(images_vector_triste)

complete_dataset = numpy.vstack((ar, ca, ci, cu, fe, hu, mi, qm, tr))

ar_correct = numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
ca_correct = numpy.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
ci_correct = numpy.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
cu_correct = numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
fe_correct = numpy.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
hu_correct = numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
mi_correct = numpy.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
qm_correct = numpy.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
tr_correct = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

ar_correct = numpy.tile(ar_correct, (len(images_vector_arbol), 1))
ca_correct = numpy.tile(ca_correct, (len(images_vector_casa), 1))
ci_correct = numpy.tile(ci_correct, (len(images_vector_circulo), 1))
cu_correct = numpy.tile(cu_correct, (len(images_vector_cuadrado), 1))
fe_correct = numpy.tile(fe_correct, (len(images_vector_feliz), 1))
hu_correct = numpy.tile(hu_correct, (len(images_vector_huevo), 1))
mi_correct = numpy.tile(mi_correct, (len(images_vector_mickey), 1))
qm_correct = numpy.tile(qm_correct, (len(images_vector_qmark), 1))
tr_correct = numpy.tile(tr_correct, (len(images_vector_triste), 1))

complete_correct = numpy.vstack((ar_correct, ca_correct, ci_correct, cu_correct, fe_correct, hu_correct, mi_correct, qm_correct, tr_correct))

# print(complete_correct.shape)

for iteration in range(NUMBER_OF_ITERATIONS):
    # feedforward the data
    IHL, HLA, OCP, predicciones = feed_forward2(complete_dataset, weight_HL, weight_OL, bias_HL, bias_OL)
    # backpropagation
    weight_OL, weight_HL = backpropagation2(complete_dataset, predicciones, weight_HL, weight_OL, 0.1, IHL, HLA, OCP, complete_correct)
    print(weight_OL)
    # print(predicciones)
    # print(weight_HL)
    # RETRY FF
    # A1, Z2, A2, Z3, A3 = feed_forward3(complete_dataset, HL, OL)

# print(complete_dataset[0])
