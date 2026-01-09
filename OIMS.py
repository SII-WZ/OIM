
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import copy
import random


def dirichlet_weighted_sq(num1, num2, data, weights=None):


    weighted_sum = data[num1] * weights[0] + data[num2] * weights[1]


    return weighted_sum ** 2


def dirichlet_weighted_sq3(num1, num2, num3, data, weights=None):


    weighted_sum = data[num1] * weights[0] + data[num2] * weights[1] + data[num3] * weights[2]

    return weighted_sum ** 2


def dirichlet_weighted(num1, num2, data, weights=None):


    return data[num1] * weights[0] + data[num2] * weights[1]


def dirichlet_weighted3(num1, num2, num3, data, weights=None):



    return data[num1] * weights[0] + data[num2] * weights[1] + data[num3] * weights[2]



choose_num=200      #the number of all photo
mix_nuw=2# the number of mix photo
sq=0


load_path="D:\zw\image_enhance\cifiar10_sq\\"


y=np.load(load_path+"pattern.npy")
x=np.load(load_path+"speckles.npy")

if sq == 1:
    sq_x = np.sqrt(x)
    np.save(load_path + 'sq_speckles', sq_x)
    print("sq sucess")
sq_x = np.load(load_path + "sq_speckles.npy")
# np.save(load_path+'spe
print(x.shape)


conca_y = []
conca_x = []


for i in range(choose_num):

    for o in range(20):
        num1 = random.randint(0, choose_num-1)
        num2 = random.randint(0, choose_num-1)

        if mix_nuw==2:
            weights = np.random.dirichlet([5,5])
            x_new = dirichlet_weighted_sq(i, num1, sq_x,weights)
            y_new = dirichlet_weighted(i, num1, y,weights)
            conca_y.append((np.array(copy.deepcopy(y_new))))
            conca_x.append((np.array(copy.deepcopy(x_new))))

        if mix_nuw==3:
            weights = np.random.dirichlet([5,5,5])
            x_new = dirichlet_weighted_sq3(i, num1, num2, sq_x,weights)
            y_new = dirichlet_weighted3(i, num1, num2, y,weights)
            conca_y.append((np.array(copy.deepcopy(y_new))))
            conca_x.append((np.array(copy.deepcopy(x_new))))


np.save(load_path+'conca_y_3bit', conca_y)
np.save(load_path+'conca_x_3bit', conca_x)
print(np.array(conca_y).shape)

