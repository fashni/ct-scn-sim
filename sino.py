#!/usr/bin/env python
#https://github.com/fashni/ct-scn-sim/blob/main/sino.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from functools import partial

import numpy as np
from skimage import io, transform as tf
from tifffile import imsave

# fungsi sinogram
def build_sinogram(imgs, crop=None):
  N = imgs.shape[0] # jumlah citra
  teta = np.linspace(0,360,N, endpoint=False)
  if crop:
    row1, row2, col1, col2 = crop
    imgs = imgs[:, row1:row2, col1:col2]
  return imgs.transpose((1, 2, 0)), teta

# fungsi inverse radon (citra 2d)
def _iradon(theta, img):
  return tf.iradon(img, theta=theta, filter_name="ramp").astype(np.float32)

# fungsi inverse radon (citra 3d)
def iradon(sinogram, theta):
  fc = partial(_iradon, theta)    # inisiasi fungsi untuk satu citra
  with ThreadPoolExecutor() as executor:
    result = executor.map(fc, sinogram)
  return np.array(list(result))

def main():
  # timer
  tt = time.time()

  # muat citra
  print('loading images...')
  img_path = 'Data/citra_kamera.tif'
  imgs = io.imread(img_path)

  # proses sinogram
  print('building sinogram...')
  sinogram, theta = build_sinogram(imgs, crop=[100, 400, 100, 400])   # crop di sini, [row1, row2, col1, col2]

  # inverse radon
  print('processing inverse radon...')
  fbps = iradon(sinogram, theta)

  # konversi hasil ke uint8
  fbps -= fbps.min()
  fbps *= 255/fbps.max()
  fbps = fbps.astype(np.uint8)

  # simpan
  print('saving sinogram...')
  imsave('Data/sino.tif', sinogram, compress=6)  # simpan sinogram
  print('saving fbp...')
  imsave('Data/fbp_ubyte.tif', fbps, compress=6) # simpan slice fbp

  # timer
  print(f'time: {time.time()-tt}s\n')

if __name__ == '__main__':
  main()
