#This file is to setup the training process.
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import kagglehub
from scripts.data_sourcing import download_data()
from scripts.train_resnet50 import train_nn

train_nn()
