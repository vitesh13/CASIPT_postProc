#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np
from scipy import spatial
import argparse
import h5py
import os

parser = argparse.ArgumentParser()

parser.add_argument('--regrid', nargs = '+',
                    help='name of file having regridded coords from Karo')
parser.add_argument('--mesh', nargs = '+',
                    help='name of file having remeshed coords from Tina')
parser.add_argument('--hdf', nargs = '+',
                    help='regridded hdf5 file from Karo')

options = parser.parse_args()

regridding_coords = np.loadtxt(options.regrid[0],skiprows=1,usecols=(0,1,2))
remeshed_coords   = np.loadtxt(options.mesh[0],skiprows=1,usecols=(0,1,2))

tree = spatial.KDTree(regridding_coords)

nbr_array = tree.query(remeshed_coords,1)[1] #finding the indices of the nearest neighbour

file_name = os.path.splitext(options.hdf[0])[0] + '_CA.hdf5'

f = h5py.File(file_name)

hdf = h5py.File(options.hdf[0])

const_values = ['C_minMaxAvg','C_volAvg','C_volAvgLastInc','F_aim','F_aimDot','F_aim_lastInc']

const_group = ['HomogState']

diff_values = ['convergedF','convergedFi','convergedFp','convergedLi','convergedLp','convergedS',' 1 _convergedStateConst']

diff_values_1 = ['F_lastInc','F']

diff_groups = ['PlasticPhases']

for i in const_values:
  f.create_dataset('/' + i,data=np.array(hdf['/' + i]))

f.create_group('HomogStates')
f.create_group('PlasticPhases')

for i in diff_values:
  if i != ' 1 _convergedStateConst' and i != 'F_lastInc':
    data_array = np.zeros((len(remeshed_coords),) + np.shape(hdf[i])[1:])
    for count,point in enumerate(nbr_array):
      data_array[count] = np.array(hdf[i][point])
    f[i] = data_array  
  else:
    data_array = np.zeros((len(remeshed_coords),) + np.shape(hdf['/PlasticPhases/' + i])[1:])
    for count,point in enumerate(nbr_array):
      data_array[count] = np.array(hdf['/PlasticPhases/' + i][point])
    f['/PlasticPhases/' + i] = data_array  
    
for i in diff_values_1:
    xsize      = int(np.max(remeshed_coords[:,0])/(remeshed_coords[1,0] - remeshed_coords[0,0])) + 1
    ysize      = int(np.max(remeshed_coords[:,1])/(remeshed_coords[1,0] - remeshed_coords[0,0])) + 1
    zsize      = int(np.max(remeshed_coords[:,2])/(remeshed_coords[1,0] - remeshed_coords[0,0])) + 1
    totalsize  = int(xsize*ysize*zsize)
    data_array = np.zeros((totalsize,) + np.shape(hdf[i])[3:])
    input_data = np.array(hdf[i]).reshape(((np.prod(np.shape(np.array(hdf[i]))[0:3]),)+np.shape(np.array(hdf[i]))[3:]))
    for count,point in enumerate(nbr_array):
      data_array[count] = input_data[point]
    data_array = data_array.reshape((zsize,ysize,xsize,) + np.shape(hdf[i])[3:])
    f[i] = data_array


