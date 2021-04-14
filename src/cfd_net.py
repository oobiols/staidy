import sys
sys.path.append('../src')
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import NS_model as NSModel
from NS_dataset import *
import argparse

keras.backend.set_floatx('float32')

parser = argparse.ArgumentParser()
# model and architecture
parser.add_argument('-m', '--model', type=int, default=0, \
                    help='0 - nsmlp, 1 - nspinn')
parser.add_argument('-l', '--architecture', type=int, nargs='*', \
                    default=[256, 256, 256, 128, 128, 128, 64, 32, 3],\
                    help='size of each layer')
parser.add_argument('-omitP', '--omitP', default=False, action='store_true')

# regularizaer coefficients
parser.add_argument('-r', '--reg', type=float, nargs='*', default=None,\
                    help='l2 regularization')
parser.add_argument('-alpha', '--alpha', type=float, nargs=3, default=[1.0, 1.0, 1.0], \
                    help='coefficients for data loss')
parser.add_argument('-beta', '--beta', type=float, nargs=3, default=[1e-4, 1e-4, 1e-4], \
                    help='coefficients for pde residual')

# sample solutions and points
parser.add_argument('-d', '--nDataPoint', type=int, default=256,\
                    help='number of data points in training')
parser.add_argument('-c', '--nCollocPoint', type=int, default=500,\
                    help='number of collocation points in training')
parser.add_argument('-s', '--nGenomePerBatch', type=int, default=64, \
                    help='number of genomes per batch')

# epochs, checkpoints
parser.add_argument('-f', '--file', default='../data/genomes_merged.h5', \
                    help='data file')
parser.add_argument('-name', '--name', default='nsPinn', help='model name prefix')
parser.add_argument('-ie', '--initTrain', type=int, default=0, \
                    help='initial train epochs')
parser.add_argument('-e', '--nEpoch', default=10000, type=int, help='epochs')
parser.add_argument('-restart', '--restart', default=False, action='store_true',\
                    help='restart from checkpoint')
parser.add_argument('-ckpnt', '--checkpoint', default=None, help='checkpoint name')

# learning rate
parser.add_argument('-lr0', '--lr0', type=float, default=5e-4, help='init leanring rate')
parser.add_argument('-lrmin', '--lrmin', type=float, default=1e-7, help='min leanring rate')
parser.add_argument('-p', '--patience', type=int, default=200, \
                    help='patience for reducing learning rate')
parser.add_argument('-lr',  '--restartLr', type=float, default=None, \
                     help='learning rate to restart training')

# save more info
parser.add_argument('-g', '--saveGradStat', default=False, action='store_true',\
                    help='save gradient statistics')

args = parser.parse_args()

archStr   = ''
l0        = args.architecture[0]
nSameSize = 1
for l in args.architecture[1:]:
  if l == l0:
    nSameSize += 1
  else:
    if nSameSize == 1:
      archStr = archStr + '-' + repr(l0)
    else:
      archStr = archStr + '-{}x{}'.format(l0, nSameSize)
    l0        = l
    nSameSize = 1
if nSameSize == 1:
  archStr = archStr + '-' + repr(l0)
else:
  archStr = archStr + '-{}x{}'.format(l0, nSameSize)

# load data
dataSet = NSDataSet()
dataSet.add_file(args.file)
dataSet.load_data()
dataSet.extract_genome_bc()
dataSet.summary()

# initial distribution of collocation points
# near uniformly following grid cells
if args.model == 1:
  dataSet.init_collocation_points(args.nCollocPoint)

# create training and validation set
nUsed = dataSet.num_genome()
nValid = int(nUsed*0.1)
nTrain = nUsed - nValid
nGPerBatch = args.nGenomePerBatch
print('{} genomes in training, {} in validation, {} per batch'.format(\
        nTrain, nValid, nGPerBatch))
# replica data generator
nDatPnt1D = np.int(np.sqrt(args.nDataPoint))
if args.model == 0:
  trainGen = dataSet.generator_bc_xy_label(\
               0, nTrain, nGPerBatch, nDataPoint1D=nDatPnt1D, \
               omitP=args.omitP)
  validGen = dataSet.generator_bc_xy_label(\
               nTrain, nUsed, nGPerBatch, nDataPoint1D=nDatPnt1D,\
               omitP=args.omitP)
  # trainGen = dataSet.generator_bcXybcRe_xy_label(0, nTrain, nGPerBatch, 1, 1)
  # validGen = dataSet.generator_bcXybcRe_xy_label(nTrain, nUsed, nGPerBatch, 1, 1)
else:
  print(nDatPnt1D)
  trainGen = dataSet.generator_uv_xy_w_label(0, nTrain, nGPerBatch, \
                                             nDataPoint1D=nDatPnt1D)
  validGen = dataSet.generator_uv_xy_w_label(nTrain, nUsed, nGPerBatch, \
                                             nDataPoint1D=nDatPnt1D)

# create model
if args.model == 0:
  modelName = args.name + archStr
  with NSModel.strategy.scope():
    nsNet = NSModel.NSModelMLP(width=args.architecture, reg=args.reg, \
                               lastLinear=True, alpha=args.alpha )
    nsNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr0))
else:
  modelName = args.name + archStr \
            + '_c{}_d{}'.format(args.nCollocPoint, args.nDataPoint)
  with NSModel.strategy.scope():
    nsNet = NSModel.NSModelPinn(width=args.architecture, reg=args.reg, \
                                alpha=args.alpha, beta=args.beta)
    nsNet.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr0))
nsNet.preview()

# callbacks
nsCB = [keras.callbacks.ModelCheckpoint(filepath='./'+modelName+'/checkpoint', \
                                        monitor='val_loss', save_best_only=True,\
                                        verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, min_delta=0.01,\
                                          patience=args.patience, min_lr=args.lrmin),
        keras.callbacks.CSVLogger(modelName+'.log', append=True)]

# load checkpoints if restart
if args.restart:
  ckpntName = modelName if args.checkpoint == None else args.checkpoint
  nsNet.load_weights(tf.train.latest_checkpoint(ckpntName))
  if args.restartLr != None:
    keras.backend.set_value(nsNet.optimizer.learning_rate, args.restartLr)

# train
nsNet.fit(trainGen, validation_data=validGen,\
          initial_epoch=args.initTrain, epochs=args.nEpoch,\
          steps_per_epoch=nTrain//nGPerBatch,\
          validation_steps=nValid//nGPerBatch,\
          verbose=2, callbacks=nsCB)
nsNet.summary()
