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
    dirname = os.path.dirname("UCF_CC_50/densities")
    if not os.path.exists(dirname):
      os.makedir(dirname)
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
      density = get_density("UCF_CC_50/densities/{}.h5".format(name), anno, np_img.shape[:-1])
      for scale in np.arange(0.6, 1.3, 0.1):
        part_size = int(225 * scale)
        for i in range(0, rows - part_size, part_size/3):
          for j in range(0, columns - part_size, part_size/3):
              part_of_image = np_img[i:(i + part_size),j:(j + part_size)]
              part_of_density = density[i:(i + part_size),j:(j + part_size)]
              resized_image = imresize(part_of_image, (225, 225))
              resized_density = imresize(part_of_density, (255, 255)) / scale / scale
              cnt += 1
              imsave("UCF_CC_50/results/{:0>5d}.jpg".format(cnt), resized_image)
              with h5py.File("UCF_CC_50/labels/{:0>5d}.h5".format(cnt), 'w') as hf:
                hf['density'] = resized_density
      print(filename + "has already processed!")
  #save_to_csv(results)
 
def load_data(imagedir='UCF_CC_50/results', densitydir='UCF_CC_50/densities'):
  filenames = get_filenames(dirname=imagedir, ext='jpg')
  np.random.shuffle(filenames)
  for filename in filenames:
    image = preprocess_data(imread("{}/{}".format(imagedir, filename)))
    with h5py.File("{}/{}.h5".format(densitydir, int(filename.split('.')[0])), 'r') as hf:
      density = np.array(hf.get('density'))

    if np.random.random() < 0.5:
      image = np.fliplr(image)
      density = np.fliplr(density)
    yield image, density
 
def preprocess_data(image):
  image[:,:,0] -= 103.939
  image[:,:,1] -= 116.779
  image[:,:,2] -= 123.68
  image /= 255.0
  return np.expand_dims(image)

if __name__ == '__main__':
  create_data()