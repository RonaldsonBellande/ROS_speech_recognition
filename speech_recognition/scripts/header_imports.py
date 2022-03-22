# Copyright © 2021 Ronaldson Bellande
from __future__ import print_function
import cv2, sys, math, random, warnings, os, os.path, json, pydicom, glob, shutil, datetime, zipfile, urllib.request, keras,  tensorflow as tf, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from os.path import basename
from PIL import Image, ImageDraw
from tensorflow import keras
from imgaug import augmenters as iaa
from tqdm import tqdm
from random import randint
import trimesh
import librosa

import nvidia_smi
from os import listdir
from xml.etree import ElementTree
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error,  classification_report, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, auc, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit, train_test_split,  KFold, cross_val_score, StratifiedShuffleSplit
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

from mrcnn import utils, visualize
# import mrcnn.model as modellib
from mrcnn.config import Config
# from mrcnn import model as modellib, utils
from mrcnn.visualize import display_images, display_instances
# from mrcnn.model import log

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from tensorflow.keras import Model

from tensorflow import convert_to_tensor
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

from keras.datasets import cifar10
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Activation, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import matplotlib.image as img

from contextlib import redirect_stdout
from multiprocessing import Pool
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

if info.free < 964157696:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.python.client import device_lib
device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']

if device_name != []:
    device_name = "/device:GPU:0"
    print("GPU")
else:
    device_name = "/device:CPU:0"
    print("CPU")


from speech_model_building import *
from speech_model_training import *
from speech_model_transfer_learning import *
from speech_model_classification import *
from deep_learning_model import *
from deep_q_learning import *
