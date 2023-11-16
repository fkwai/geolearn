import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
from hydroDL import kPath
from hydroDL.model import rnn, crit, trainBasin
import torch
import time
from hydroDL.master import basinFull, slurm, dataTs2End
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers, optimizers


# load old data
outFile = '/home/kuai/work/VegetationWater/data/model/data/testData.npz'
dataOld = np.load(outFile)
xTrain = dataOld['xTrain']
xTest = dataOld['xTest']
yTrain = dataOld['yTrain']
yTest = dataOld['yTest']
dataTup1 = (xTrain, None, None, yTrain)
dataTup2 = (xTest, None, None, yTest)
outFolder = os.path.join(kPath.dirVeg, 'model', 'old')
if not os.path.exists(outFolder):
    os.mkdir(outFolder)

[nx, nxc, ny, nyc, nt, ns] = trainBasin.getSize(dataTup1)


def build_model(input_shape):
    DROPOUT = 0.05
    LOSS = 'mse'
    Areg = regularizers.l2(1e-5)
    Breg = regularizers.l2(1e-3)
    Kreg = regularizers.l2(1e-10)
    Rreg = regularizers.l2(1e-15)
    model = Sequential()
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                  return_sequences=True, \
                  bias_regularizer= Breg))
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                    return_sequences=True, \
                    bias_regularizer= Breg))
    model.add(LSTM(10, input_shape=input_shape, dropout = DROPOUT,recurrent_dropout=DROPOUT,\
                   bias_regularizer= Breg))
    model.add(Dense(1))
#    optim = optimizers.SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)
#    optim = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
    model.compile(loss=LOSS, optimizer='Nadam')
    # fit network
    return model


model = build_model((xTrain.shape[1], xTrain.shape[2]))
EPOCHS = int(5e3)
BATCHSIZE = int(2e5)
history = model.fit(
    xTrain, yTrain, epochs=EPOCHS, batch_size=BATCHSIZE, verbose=1, shuffle=False,
    validation_data=(xTest, yTest)
)
yP= model.predict(xTest)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
ax.plot(yP[:, 0], yTest, '*')
from hydroDL import utils

rmse, corr = utils.stat.calErr(yP[:, 0],yTest)
corr
rmse

import dill
dill.dump_session('temp.db')

dill.detect.errors(dataOld)
