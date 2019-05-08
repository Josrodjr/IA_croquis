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




What Scott said works for me,
For the other article (Previous Neural Net) I added the code:

print('weights = '+str(weights))
print('bias = '+str(bias))

And then copied the whole answer into the top of the new code:

import numpy as np
weights = [[ -1.79848303], #Pasted in (Make sure to add commas betwene)
[ 10.37194106],
[ 10.37191014],
[ -4.22269027]]
bias = [-0.72913676] #Pasted in

def sigmoid(x): 
return 1/(1+np.exp(-x))


def test(x,y,z,j):
single_point = np.array([x,y,z,j])
result = sigmoid(np.dot(single_point, weights) + bias)
print(result)


test(0,0,1,0)

# LO VERDADERAMENTE SAD
feedforward, backpropagation, weight_update, sigmoid

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


lambda es lo que vamos a cambiar basicamente

falta un parcial y un proyecto
entrega de proyecto 9 mayo
parcial 16 mayo
final 3 junio

pocas capas intermedias = bias


modelos con mas overfit lambda mas peque;ab
modelos con menos ooverfil lambda mas pequ;abs