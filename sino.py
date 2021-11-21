#!/usr/bin/env python
#https://github.com/fashni/ct-scn-sim/blob/main/sino.py
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from functools import partial

import numpy as np
from skimage import io, transform as tf

# fungsi sinogram
def build_sinogram(imgs):
  N = imgs.shape[0] # jumlah citra
  teta = np.linspace(0,360,N, endpoint=False)
  # crop di sini, imgs = imgs[:, row1:row2, col1:col2]
  imgs = imgs[:, 100:400, 100:400]
  return imgs.transpose((1, 2, 0)), teta

# fungsi inverse radon (citra 2d)
def _iradon(theta, img):
  return tf.iradon(img, theta=theta, filter_name="ramp").astype(np.float32)

# fungsi inverse radon (citra 3d)
def iradon(sinogram, theta):
  fc = partial(_iradon, theta) # inisiasi fungsi untuk satu citra
  with ThreadPoolExecutor() as executor:
    result = executor.map(fc, sinogram)
  return np.array(list(result))

def main():
  # timer
  tt = time.time()

  print('loading images...')
  img_path = 'Data/citra_kamera.tif'
  imgs = io.imread(img_path)

  print('building sinogram...')
  sinogram, theta = build_sinogram(imgs) # panggil fungsi
  io.imsave('Data/sino.tif', sinogram) # simpan sinogram

  print('processing inverse radon...')
  fbps = iradon(sinogram, theta)

  # konversi ke uint8 dan simpan
  fbps -= fbps.min()
  fbps *= 255/fbps.max()
  fbps = fbps.astype(np.uint8)
  io.imsave('Data/fbp_ubyte.tif', fbps)

  # timer
  print(f'time: {time.time()-tt}s\n')

if __name__ == '__main__':
  main()
