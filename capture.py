#!/usr/bin/env python
#https://github.com/fashni/ct-scn-sim/blob/main/capture.py
import time
import numpy as np
import RPi.GPIO as GPIO
from picamera import PiCamera
from skimage.color import rgb2gray
from skimage import io, img_as_ubyte
from tifffile import imsave

N = 256 # jumlah citra yg akan diambil
IMG_DIM = (512, 512) # ukuran dimensi citra

# inisiasi GPIO (motor)
GPIO.setmode(GPIO.BOARD)
control_pins = [7,11,13,15]
for pin in control_pins:
  GPIO.setup(pin, GPIO.OUT)
  GPIO.output(pin, 0)
halfstep_seq = [
  [1,0,0,0],
  [1,1,0,0],
  [0,1,0,0],
  [0,1,1,0],
  [0,0,1,0],
  [0,0,1,1],
  [0,0,0,1],
  [1,0,0,1]
]

# inisiasi kamera
camera = PiCamera()
camera.resolution = IMG_DIM
camera.start_preview()
time.sleep(2)

# inisiasi array citra 3d kosong
cit3d = np.empty(tuple([N] + list(IMG_DIM)), dtype=np.uint8)

# mulai proses
tt = time.time()
step = 512//N
for i in range(512):
  # ambil citra
  if i%step == 0:
    output = np.empty(tuple(list(IMG_DIM)+[3]), dtype=np.uint8)
    camera.capture(output, 'rgb')
    img_gray = rgb2gray(output)
    np.copyto(cit3d[i//step], img_as_ubyte(img_gray))
    print(f'done {1 + i//step}/{N}')

  # putar stepper motor
  for halfstep in range(8):
    for pin in range(4):
       GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
    time.sleep(0.01)

# bersih-bersih
camera.stop_preview()
GPIO.cleanup()

# simpan citra 3d
print('Saving images...')
imsave('Data/citra_kamera.tif', cit3d, compress=6)
print(f'time: {time.time()-tt}s\n')
