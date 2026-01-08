
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import copy
import random
def add_sq_turn(num1,num2,data):

    result=(data[num1]/2+data[num2]/2)*(data[num1]/2+data[num2]/2)
    return result
def add_sq_turn3(num1,num2,num3,data):

    result=(data[num1]/3+data[num2]/3+data[num3]/3)*(data[num1]/3+data[num2]/3+data[num3]/3)
    return result
def add_turn(num1,num2,data):

    result=data[num1]/2+data[num2]/2
    return result
def add_turn3(num1,num2,num3,data):

    result=data[num1]/3+data[num2]/3+data[num3]/3
    return result




choose_num=200      #the number of all photo
mix_nuw=3# the number of mix photo
sq=0


load_path="D:\\OIMS\\"


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

    for o in range(10):
        num1 = random.randint(0, choose_num-1)
        num2 = random.randint(0, choose_num-1)

        if mix_nuw==2:
            x_new = add_sq_turn(i, num1, sq_x)
            y_new = add_turn(i, num1, y)
            conca_y.append((np.array(copy.deepcopy(y_new))))
            conca_x.append((np.array(copy.deepcopy(x_new))))

        if mix_nuw==3:
            x_new = add_sq_turn3(i, num1, num2, sq_x)
            y_new = add_turn3(i, num1, num2, y)
            conca_y.append((np.array(copy.deepcopy(y_new))))
            conca_x.append((np.array(copy.deepcopy(x_new))))


np.save(load_path+'conca_y_3bit', conca_y)
np.save(load_path+'conca_x_3bit', conca_x)
print(np.array(conca_y).shape)

