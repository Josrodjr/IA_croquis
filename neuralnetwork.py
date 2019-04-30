import numpy 
import matplotlib.pyplot as pyplot
import pickle

# local library for the neuralnetwork methods
from neural_lib import images_to_pickle, feedforward, backpropagation, weight_update, sigmoid

# set a randomseed
numpy.random.seed(69)

# numpy.seterr(divide='ignore', invalid='ignore')
# numpy.warnings.filterwarnings('ignore')

# ******************************* CIRCULO ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Circle_new"
savename = f"circulo.p"

images_to_pickle(mypath, savename)
images_vector_circulo = pickle.load(open(savename, "rb"))

# ******************************* CUADRADO ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Square_new"
savename = f"cuadrado.p"

images_to_pickle(mypath, savename)
images_vector_cuadrado = pickle.load(open(savename, "rb"))

# ******************************* HUEVO ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Egg_new"
savename = f"huevo.p"

images_to_pickle(mypath, savename)
images_vector_huevo = pickle.load(open(savename, "rb"))

# ******************************* ARBOL ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Tree_new"
savename = f"arbol.p"

images_to_pickle(mypath, savename)
images_vector_arbol = pickle.load(open(savename, "rb"))

# ******************************* CASA ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\House_new"
savename = f"casa.p"

images_to_pickle(mypath, savename)
images_vector_casa = pickle.load(open(savename, "rb"))

# ******************************* FELIZ ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Smile_new"
savename = f"feliz.p"

images_to_pickle(mypath, savename)
images_vector_feliz = pickle.load(open(savename, "rb"))

# ******************************* TRISTE ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Sad_new"
savename = f"triste.p"

images_to_pickle(mypath, savename)
images_vector_triste = pickle.load(open(savename, "rb"))

# ******************************* INTERR ************************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Question_new"
savename = f"qmark.p"

images_to_pickle(mypath, savename)
images_vector_qmark = pickle.load(open(savename, "rb"))

# ******************************* MICKEY ***********************************
# get all files in a path
mypath = r"C:\Users\Josro\Desktop\training_all\Mickey_new"
savename = f"mickey.p"

images_to_pickle(mypath, savename)
images_vector_mickey = pickle.load(open(savename, "rb"))


# ***************************** ACTUAL NN **********************************
