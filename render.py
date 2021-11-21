#!/usr/bin/env python
#https://github.com/fashni/ct-scn-sim/blob/main/render.py
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure
from scipy.ndimage import zoom
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# fungsi ekstraksi data mesh permukaan 2d dari citra 3d
def gen_surface(imgs, scale=1, level=None, step=1):
  imgs = imgs[::-1].transpose((1, 2, 0))                        # konversi koordinat (z, y, x) ke (y, x, z)
  if scale == 0:
    raise ValueError('Scale value must be greatr than 0')
  if scale > 0 and scale < 1:
    imgs = zoom(imgs, scale)
  return measure.marching_cubes(imgs, level=level, step_size=step)

# fungsi render objek 3d
def render(verts, faces, normals, axlim=200, edge_color=None, face_color=None, opacity=1):
  # https://stackoverflow.com/a/56869214/14402153
  mesh = Poly3DCollection(verts[faces], alpha=opacity)
  ls = LightSource(azdeg=225.0, altdeg=45.0)

  normalsarray = np.array([np.array((np.sum(normals[face[:], 0]/3), np.sum(normals[face[:], 1]/3), np.sum(normals[face[:], 2]/3))/np.sqrt(np.sum(normals[face[:], 0]/3)**2 + np.sum(normals[face[:], 1]/3)**2 + np.sum(normals[face[:], 2]/3)**2)) for face in faces])

  min_ = np.min(ls.shade_normals(normalsarray, fraction=1.0))   # min shade value
  max_ = np.max(ls.shade_normals(normalsarray, fraction=1.0))   # max shade value
  diff = max_-min_
  newMin = 0.3
  newMax = 0.95
  newdiff = newMax-newMin

  if face_color is None:
    face_color = np.random.rand(3,)
  face_color = np.array(face_color)
  face_color = np.array([face_color*(newMin + newdiff*((shade-min_)/diff)) for shade in ls.shade_normals(normalsarray, fraction=1.0)])

  mesh.set_facecolor(face_color)
  mesh.set_edgecolor(edge_color)

  fig = plt.figure(figsize=(5, 5))
  ax = fig.add_subplot(111, projection='3d')
  ax.add_collection3d(mesh)
  try:
    xlim, ylim, zlim = axlim
  except:
    xlim = ylim = zlim = axlim
  ax.set_xlim(0, xlim)
  ax.set_ylim(0, ylim)
  ax.set_zlim(0, zlim)

  plt.tight_layout()
  plt.show()

def main():
  # muat citra
  print('loading images...')
  imgs = io.imread('Data/fbp_ubyte.tif')

  # menentukan nilai pixel objek otomatis (asumsi: nilai pixel objek = nilai pixel paling banyak)
  bins = 256 if imgs.dtype==np.uint8 else 1000
  hist, bins_edge = np.histogram(imgs, bins=bins)
  px = np.linspace(imgs.min(), imgs.max(), bins)
  level = px[hist.argmax()]                                     # nilai pixel permukaan objek
  print(f'object surface pixel value: {level}')

  # hitung data mesh permukaan
  print('generating surface mesh...')
  scale = 0.25                                                  # skala citra antara 0-1 (diperkecil)
  v, f, n, val = gen_surface(imgs, scale=scale, level=level)

  # render
  print('rendering...')
  render(v, f, n, axlim=imgs.shape[0]*scale+10, opacity=.9)


if __name__ == '__main__':
  main()
