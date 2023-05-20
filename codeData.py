from os import listdir
from numpy import asarray
from numpy import save
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

folder = 'data_AI/'
photos, labels = list(), list()
for file in listdir(folder):
  if file.startswith('bicycle'):
    output = 0
  elif file.startswith('boat'):
    output = 1
  elif file.startswith('car'):
    output = 2
  elif file.startswith('motorbike'):
    output = 3
  elif file.startswith('airplane'):
    output = 4
  elif file.startswith('train'):
    output = 5
  elif file.startswith('truck'):
    output = 6
  else:
    continue
  photo = load_img(folder + file, target_size=(90, 120))
  photo = img_to_array(photo)
  photos.append(photo)
  labels.append(output)
photos = asarray(photos)
labels = asarray(labels)
save('photos_AI.npy', photos)
save('labels_AI.npy', labels)

