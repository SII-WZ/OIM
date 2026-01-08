import xlwt
import random
import os
import numpy as np
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.metrics import accuracy_score, f1_score,mean_squared_error,mean_absolute_error
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras
import tensorflow as tf
import xlrd
import xlutils
import copy
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from skimage.metrics import structural_similarity as compare_ssim
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
import numpy
print("begin")



def createCNN(outsize, input_shape):
    model_input = Input(shape=input_shape)

    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(model_input)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x) #

    x = Conv2D(16, kernel_size=(3, 3), strides=2, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x)


    x = Flatten()(x)              #变成一维数组
    x = Dense(outsize * outsize, activation='sigmoid')(x)

    cnn = Model(inputs=model_input, outputs=x)
    return cnn
def elvluate(labels, pred):
    corr = []
    mse = []
    ssim = []
    correct=[]
    pred=numpy.array(pred)
    labels = numpy.array(labels)

    for i in range(0, pred.shape[0]):
        c1 = np.corrcoef(labels[i], pred[i])[0, 1]
        p1 = pred[i].reshape(ini_outsize, ini_outsize)
        y1 = pred[i].reshape(ini_outsize, ini_outsize).astype(p1.dtype)
        m1 = compare_mse(y1, p1)
        if outsize < 7:
            y1 = y1.repeat(2, axis=1).repeat(2, axis=0)
            p1 = p1.repeat(2, axis=1).repeat(2, axis=0)
        s1 = compare_ssim(y1, p1, data_range=p1.max() - p1.min())
        corr.append(c1)
        mse.append(m1)
        ssim.append(s1)
    pred[pred>= 0.5] = 1
    pred[pred < 0.5] = 0
    for i in range(0, pred.shape[0]):
        a1 = 1 - np.mean(abs(labels[i] - pred[i]))
        correct.append(a1)

    return [np.mean(correct), np.mean(corr)]#,np.mean(mse),  np.mean(ssim)

def test_step(model, images, labels):
    pred = model(images)         #  loss_step = keras.losses.binary_crossentropy(labels, pred)  test_loss(loss_step)
    evaluationC=c_evaluate(labels, pred)
    return evaluationC
# TRAIN
def train(epoch_num,model,train_dataset,test_dataset):
    for epoch in range(epoch_num):
        acc=[]
        progress=0
        for (images, labels) in train_dataset:
            progress=progress+1
            step=(progress*global_batch_size/ini_data_long)
            print("\r{:^3.0f}%".format(step*100), end= '')
            train_step(model, images, labels)
        for (images, labels) in test_dataset:
            acc.append(test_step(model, images, labels))
        acc = np.array(acc)
        print("Epoch {} train loss is {},  test acc is {}".format(epoch,train_loss.result(),np.mean(acc[0,])))
        train_loss.reset_states()
    return acc

def c_evaluate(y_actual, y_predicted):
    y_actual=np.array(y_actual)
    y_predicted=np.array(y_predicted)
    accuracy = []
    f1score = []
    for i in range(0, y_actual.shape[0]):

       # a = mean_squared_error(y_actual[i], y_predicted[i])
     #   a = compare_ssim(y_actual[i,], y_predicted[i,],data_range=y_actual[i,].max() - y_actual[i,].min())
     #   a = 1 - mean_squared_error(y_actual[i], y_predicted[i])
        a = 1 - mean_absolute_error(y_actual[i], y_predicted[i])
        accuracy.append(a)
    print(y_actual.shape)

    return [np.mean(accuracy), np.mean(accuracy)]
def acc_test(model,test_dataset):
    acc = []
    accmap = []
    predict=[]
    for (images, labels) in test_dataset:
        acc.append(test_step(model, images, labels))
#        accmap.append(c_evaluate(model, images, labels)[0])
        predict.extend(model(images))
    acc = np.array(acc)
    print("test acc is {}".format(np.mean(acc)))
    return acc, accmap,predict
def test(epoch_num,model,test_dataset):
    acc = []
    for epoch in range(epoch_num):
        for (images, labels) in test_dataset:
            acc.append(test_step(model, images, labels))
        print("Epoch {} train loss is {},  test acc is {}".format(epoch, train_loss.result(), np.mean(acc[0,])))
        train_loss.reset_states()
    return acc

def train_step(model, images, labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step =keras.losses.binary_crossentropy(labels,pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))




    train_loss(loss_step)


def get_callbacks():
    #     checkpoint = ModelCheckpoint(best_model_path, monitor='val_loss', verbose=1,
    #                                  save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=10, verbose=2, mode='min')
    #     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
    #                                   verbose=1, mode='auto', cooldown=0, min_lr=1e-7)

    csv_logger = CSVLogger(load_path + 'model_enhancelog.csv', append=True)

    #     return [checkpoint, csv_logger,
    #             reduce_lr, early_stopping
    #             ]
    return [csv_logger]




optimizer = keras.optimizers.Adadelta(lr=1e-1)
ada = keras.optimizers.Adadelta(lr=1e-1)
oims=1 #1 is use oims 0 is without oims


load_path="D:\\OIMS\\"
y_meta=np.load(load_path+"y.npy")/255
x_meta=np.load(load_path+"x.npy")
y_train = np.load( load_path+'conca_x_3bit.npy')
x_train = np.load(load_path+"conca_y_3bit.npy")
y_test = y_meta[-100:,]
x_test = x_meta[-100:,]
y_train_01 = y_meta[:200,]
x_train_01 = x_meta[:200,]
ini_outsize=int(np.sqrt(y_train_01.shape[1]))

imagesize = x_test.shape[1]    #输出行数量
outsize = np.sqrt(y_test.shape[1])
input_shape = (imagesize, imagesize,1)

if oims ==1:
    load_path=load_path+"oims\\"
if oims ==0:
    load_path = load_path + "witout_oims\\"

if not os.path.exists(load_path):
    os.makedirs(load_path)


global_batch_size=16



train_loss = tf.keras.metrics.Mean('train_loss')
test_loss = tf.keras.metrics.Mean('test_loss')

print(x_train.shape)
model =createCNN2(outsize=outsize, input_shape=input_shape)
print(y_train.shape)

#model=load_model('pretrain_clf3.hdf5')

model.compile(loss=keras.losses.mean_squared_error, optimizer=ada,metrics=['mean_squared_error',"mean_absolute_error"])
if oims==1:
    print("work  on OMIS")
    print(x_train.shape)
    model.fit(x_train, y_train, batch_size=32, epochs=200,callbacks=get_callbacks(),   validation_data=(x_test, y_test))  #pretrain
    model.fit(x_train_01, y_train_01, batch_size=32, epochs=2000, callbacks=get_callbacks(),               #finetune
               validation_data=(x_test, y_test))
else :
    print("work on no OMIS")
    print(x_train_01.shape)
    model.fit(x_train_01, y_train_01, batch_size=32, epochs=2000,callbacks=get_callbacks(),   validation_data=(x_test, y_test))
#save_model(model,load_path+ 'self' + '.hdf5')
accy=[]
accymap=[]
predcit=[]
for i in  range(1):
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(100)

    accy1,accymap1,predcit1=acc_test (model, test_dataset)
    accy.extend(accy1)
    accymap.extend(accymap1)
    predcit.extend(predcit1)
accy=np.array(accy)
accyout=[]
if oims ==0:
    np.save(load_path+"predict.npy",np.array(predcit))
else:
    np.save(load_path+"model_oims_predict.npy",np.array(predcit))

for i in range(int(len(accy))):
    print(np.mean(accy[i:i+1,0]))
    accyout.append(np.mean(accy[i:i+1,0]))

wk = xlwt.Workbook()
sh = wk.add_sheet("shee1")
for i in range(len(accyout)):
    sh.write(i+1, 0, 1*accyout[int(i)])

wk.save(load_path+"model_oims.xls")
