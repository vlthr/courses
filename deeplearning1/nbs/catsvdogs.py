from theano.sandbox import cuda
cuda.use('gpu0')
import os, sys
from __future__ import division, print_function
import math

LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/data/dogscatsredux'
path = DATA_HOME_DIR + '/' #'/sample/'
test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
model_path=DATA_HOME_DIR + '/models/'
train_path=path + '/train/'
valid_path=path + '/valid/'
#Allow relative imports to directories above lesson1/
sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.chdir("../private/dogsvcatsredux/")
current_dir = os.getcwd()

import utils; reload(utils)
from utils import *
import vgg16
import vgg16bn

batch_size=48
no_of_epochs=10
