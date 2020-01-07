#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np

data = []
with open('resMDRX.3D.geom','r') as f:
  for i, line in enumerate(f):
    if i < 1:
      data.append(line)

dummy = data[0].split()

print(dummy)
if int(dummy[-1]) <= 1:
  n_elem = int(dummy[2])*int(dummy[4])
else:
  n_elem = int(dummy[2])*int(dummy[4])*int(dummy[6])

data_to_geom = []
data_to_geom.append('{} header'.format(1))  #so that we can modify this later
data_to_geom.append('grid a {} b {} c {}'.format(dummy[2],dummy[4],dummy[6]))
data_to_geom.append('size x {} y {} z {}'.format(int(dummy[2])*4,int(dummy[4])*4,int(dummy[6])*4))  #give input that will kind of decide the spacing (like in DREAM3D)
    
data_to_geom.append('origin x 0.0 y 0.0 z 0.0')
data_to_geom.append('homogenization 1')

geom_data_1 = np.loadtxt('resMDRX.texture_MDRX.txt',skiprows=0,usecols=(2))
data_to_geom.append('microstructures {}'.format(int(np.amax(geom_data_1))+1))

#write microstructure part
data_to_geom.append('<microstructure>')
for count,grain in enumerate(geom_data_1):
  data_to_geom.append('[Grain{:02d}]'.format(int(grain)+1))
  data_to_geom.append('crystallite 1')
  data_to_geom.append('(constituent) phase 1 texture {} fraction 1.0'.format(int(grain)+1))

#write texture part
data_to_geom.append('<texture>')
with open('resMDRX.texture_MDRX.txt') as m:
  texture_data = m.readlines()

for i,texture in enumerate(texture_data):
  data_to_geom.append('[Grain{:02d}]'.format(i+1))
  orientations = texture.split()
  data_to_geom.append('(gauss) phi1 {} Phi {} phi2 {} scatter 0.0 fraction 1.0'.format(orientations[4],orientations[6],orientations[8]))

# calculating header length
header_value = len(data_to_geom) - 1
data_to_geom[0] = '{} header'.format(header_value)
geom_data = np.loadtxt('resMDRX.3D.geom',skiprows=1,usecols=(1))
numbers = geom_data + 1
#write numbers in geom file 
for i in numbers:
  data_to_geom.append(int(i))

for line in data_to_geom:
  print(line)

array = np.array(data_to_geom)
np.savetxt('test.geom',array,fmt='%s',newline='\n') 
