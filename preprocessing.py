import os
import re
import numpy as np
import csv
import h5py
import sys
import scipy
import scipy.ndimage
from scipy.misc import imresize, imsave, imread
from numpy.linalg import norm
from scipy.io import loadmat

def get_filenames(dirname = 'UCF_CC_50', ext='mat'):
  r = re.compile("[\w]+.{}$".format(ext))
  return filter(r.match, os.listdir(dirname))
 
def get_annolist(filename):
  return loadmat(filename)['annPoints']

def get_density(filename, anno, imshape):
  if os.path.isfile(filename):
    with h5py.File(filename, 'r') as hf:
      density = np.array(hf.get('density'))
  else:
    density = create_density_map(anno, imshape)
    with h5py.File(filename, 'w') as hf:
      hf['density'] = density
  return density

def create_density_map(annos, imshape):
  gt = np.zeros(imshape, dtype='uint8')
  
  for dot in annos:
    try:
      gt[int(dot[1]), int(dot[0])] = 1
    except IndexError:
      print dot[1], dot[0], sys.exc_info()

  density = np.zeros(gt.shape, dtype=np.float32)
  gt_count = np.count_nonzero(gt)
  if gt_count == 0:
      return density
  pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
  leafsize = 2048
  tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
  distances, locations = tree.query(pts, k=2, eps=10.)
  for i, pt in enumerate(pts):
    pt2d = np.zeros(gt.shape, dtype=np.float32)
    pt2d[pt[1],pt[0]] = 1.
    if gt_count > 1:
        sigma = distances[i][1]
    else:
        sigma = np.average(np.array(gt.shape))/2.
    density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
  return density
 
def create_data(dirname='UCF_CC_50'):
  filenames = get_filenames()
  cnt = 0
  for filename in filenames:
      anno = get_annolist("{}/{}".format(dirname,filename))
      name = '.'.join(filename.split('.')[:-1])
      np_img = imread("{}/{}.jpg".format(dirname, os.path.splitext(filename)[0].split('_')[0]), mode='RGB')
      rows, columns, _= np_img.shape

      densities_path = "{}/densities".format(dirname)
      if not os.path.exists(densities_path):
        os.makedirs(densities_path)
      results_path = "{}/results".format(dirname)
      if not os.path.exists(results_path):
        os.makedirs(results_path)
      labels_path = "{}/labels".format(dirname)
      if not os.path.exists(labels_path):
        os.makedirs(labels_path)

      density = get_density("{}/{}.h5".format(densities_path, name), anno, np_img.shape[:-1])
      for scale in np.arange(0.6, 1.3, 0.1):
        part_size = int(225 * scale)
        for i in range(0, rows - part_size, part_size/3):
          for j in range(0, columns - part_size, part_size/3):
              part_of_image = np_img[i:(i + part_size),j:(j + part_size)]
              part_of_density = density[i:(i + part_size),j:(j + part_size)]
              resized_image = imresize(part_of_image, (225, 225))
              resized_density = imresize(part_of_density, (225, 225)) / scale / scale
              cnt += 1
              imsave("{}/{:0>5d}.jpg".format(results_path, cnt), resized_image)
              with h5py.File("{}/{:0>5d}.h5".format(labels_path, cnt), 'w') as hf:
                hf['density'] = resized_density
      print(filename + " has already processed!")
  #save_to_csv(results)
 
def load_data(batch_size = 64, imagedir='UCF_CC_50/results', densitydir='UCF_CC_50/labels'):
  filenames = get_filenames(dirname=imagedir, ext='jpg')
  np.random.shuffle(filenames)
  images, densities = np.zeros([batch_size, 225, 225, 3]), np.zeros([batch_size, 225, 225, 1])
  #import ipdb; ipdb.set_trace()
  for i, filename in enumerate(filenames):
    image = preprocess_data(imread("{}/{}".format(imagedir, filename)))
    with h5py.File("{}/{}.h5".format(densitydir, filename.split('.')[0]), 'r') as hf:
      density = np.array(hf.get('density'))
    if np.random.random() < 0.5:
      image = np.fliplr(image)
      density = np.fliplr(density)
    density = np.expand_dims(density, 2)
    images[i % batch_size] = image
    densities[i % batch_size] = density
    if (i + 1) % batch_size == 0:
      yield np.array(images), np.array(densities)
      images, densities = np.zeros([batch_size, 225, 225, 3]), np.zeros([batch_size, 225, 225, 1])
 
def preprocess_data(image):
  image = image.astype('float32')
  image[:,:,0] -= 103.939
  image[:,:,1] -= 116.779
  image[:,:,2] -= 123.68
  image /= 255.0
  return image

if __name__ == '__main__':
  create_data()