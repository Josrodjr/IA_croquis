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