import os
import re
import glob
import math
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class DataProcessor():
  def __init__(self, path='./input/preprocessed', data_dir='./input/data', imw=128, imh=128, z_threshold=-3.5):
    self.path = path
    self.data_dir = data_dir
    self.imsize = (imw, imh)
    self.z_threshold = z_threshold
    return

  def gen_dataset(self):
    if os.path.isdir(self.path):
      # Get data from train/test file dir
      train_len = len(os.listdir(f"{self.path}/train"))
      test_len = len(os.listdir(f"{self.path}/test"))
      data = {
        'x_train': np.empty((train_len, self.imsize[0], self.imsize[1])),
        'y_train': np.empty((train_len)),
        'x_test': np.empty((test_len, self.imsize[0], self.imsize[1])),
        'y_test': np.empty((test_len))
      }

      for f in glob.glob(f"{self.path}/*/*.png"):
        res = re.findall(r"(\w+)\\c(\d+)_(\d+)\.png$", f)[0]
        
        img = np.asarray(Image.open(f).convert('L'))
        data[f"x_{res[0]}"][int(res[2])] = img
        data[f"y_{res[0]}"][int(res[2])] = int(res[1])
      
      self.x_train = data['x_train']
      self.y_train = data['y_train']
      self.x_test = data['x_test']
      self.y_test = data['y_test']
    else:
      os.makedirs(self.path, exist_ok=True)
      
      # Collect dataset
      data = []
      for f in glob.iglob(f"{self.data_dir}/*/*.png"):
        label = os.path.split(f)[1].split('-')[0]
        coded = self.__binary_encode(label)
        image = self.__read_image(f)
        data.append({'source': label, 'label': coded, 'img': image, 'mean': np.mean(image)})
      
      data = pd.DataFrame.from_dict(data)

      # Remove outliers
      if self.z_threshold:
        data = self.__remove_outliers(data)
      
      # Balance dataset
      data = self.__min_sample_by_source(data)

      # Train/Test split
      x = np.asarray([list(x) for x in data['img'].values])
      y = np.asarray(data['label'].values)

      x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
      self.x_train = x_train
      self.y_train = y_train
      self.x_test = x_test
      self.y_test = y_test

      # Save to dir
      splits = {'train': [x_train, y_train], 'test': [x_test, y_test]}
      for split in splits:
        os.makedirs(f"{self.path}/{split}/", exist_ok=True)
        
        dataset = splits[split][0]
        labels = splits[split][1]
        for i, img in enumerate(dataset):
          Image.fromarray(img).save(f"{self.path}/{split}/c{labels[i]}_{i}.png")

    # Prepare for keras
    sample = self.x_train[0]
    self.channels = len(sample.shape) - 1
    self.classes = len(np.unique(np.array(self.y_train))) - 1

    self.r_train = self.y_train.copy()
    self.r_test  = self.y_test.copy()

    self.x_train = self.x_train.reshape(self.x_train.shape[0], self.channels, self.imsize[0], self.imsize[1]).astype('float32') / 255
    self.y_train = np.asarray(self.y_train).astype('int32').reshape((-1, 1))

    self.x_test = self.x_test.reshape(self.x_test.shape[0], self.channels, self.imsize[0], self.imsize[1]).astype('float32') / 255
    self.y_test = np.asarray(self.y_test).astype('int32').reshape((-1, 1))

  def get_train_data(self):
    return self.x_train, self.y_train
  
  def get_test_data(self):
    return self.x_test, self.y_test

  def __binary_encode(self, target):
    return 1 if target == 'COVID' else 0

  def __read_image(self, f):
    image = np.asarray(Image.open(f).resize(self.imsize, Image.ANTIALIAS).convert('L'))
    return image
  
  def __remove_outliers(self, data):
    data['zscore'] = stats.zscore(data['mean'], ddof=0)
    data = data.loc[data['zscore'] > self.z_threshold].reset_index(drop=True)
    
    return data
  
  def __min_sample_by_source(self, data):
    min_index = np.argmin(data['label'].value_counts())
    covid_pos = shuffle(data[data['label'] == 1]).reset_index(drop=True)
    covid_neg = shuffle(data[data['label'] == 0]).reset_index(drop=True)

    max_class = [covid_pos, covid_neg][min_index]
    min_class = [covid_neg, covid_pos][min_index]
    num_min_samples = len(min_class)

    sample_size = math.ceil(int(num_min_samples / len(pd.unique(max_class['source']))))
    max_class = max_class.groupby('source').apply(lambda x: x.sample(sample_size)).reset_index(drop=True)
    data = pd.concat([max_class[:min(len(max_class), num_min_samples)], min_class[:min(len(min_class), num_min_samples)]]).reset_index(drop=True)
    
    return data