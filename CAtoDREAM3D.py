#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np
import h5py
import math

class AttributeManagerNullterm(h5py.AttributeManager): 
  """
  Attribute management for DREAM.3D hdf5 files.
  
  String attribute values are stored as fixed-length string with NULLTERM
  
  References
  ----------
    https://stackoverflow.com/questions/38267076
    https://stackoverflow.com/questions/52750232

  """ 

  def create(self, name, data, shape=None, dtype=None):
    if isinstance(data,str):
      tid = h5py.h5t.C_S1.copy()
      tid.set_size(len(data + ' '))
      super().create(name=name,data=data+' ',dtype = h5py.Datatype(tid))
    else:
      super().create(name=name,data=data,shape=shape,dtype=dtype)
     

h5py._hl.attrs.AttributeManager = AttributeManagerNullterm # 'Monkey patch'

#--------------------------------------------------------------------------
Crystal_structures = {'fcc': 1,
                      'bcc': 1,
                      'hcp': 0,
                      'bct': 7,
                      'ort': 6} #TODO: is bct Tetragonal low/Tetragonal high?
Phase_types = {'Primary': 0} #further additions to these can be done by looking at 'Create Ensemble Info' filter
#--------------------------------------------------------------------------
#Build array of euler angles for each cell
#--------------------------------------------------------------------------
#orientation_array = np.loadtxt('resMDRX.texture_MDRX.txt',skiprows=0,usecols=(4,6,8)) # ID phi1 phi phi2
orientation_array = np.loadtxt('resMDRX..ang',skiprows=0,usecols=(0,1,2)) # ID phi1 phi phi2
#orientation_array = orientation_array*(180.0/math.pi)

geom_number = np.loadtxt('resMDRX.3D.geom',dtype=int,skiprows=1,usecols=(1))

grid = []
with open('resMDRX.3D.geom','r') as f:
  for i, line in enumerate(f):
    if i < 1:
      grid.append(line)

dummy = [grid[0].split()[2], grid[0].split()[4], grid[0].split()[6]]
dummy = [int(i) for i in dummy]

#orientation_data = np.zeros((len(geom_number),3))
orientation_data = orientation_array

#for count,i in enumerate(geom_number):
#  orientation_data[count] = orientation_array[i]

#--------------------------------------------------------------------------
o = h5py.File('CA_output.dream3D','w')
o.attrs['DADF5toDREAM3D'] = '1.0'
o.attrs['FileVersion']    = '7.0' 

for g in ['DataContainerBundles','Pipeline']: # empty groups (needed)
  o.create_group(g)

data_container_label = 'DataContainers/ImageDataContainer'        
cell_data_label      = data_container_label + '/CellData'

o[cell_data_label + '/Phases'] = np.ones(tuple(dummy)+(1,),dtype=np.int32) 

# Data eulers
orientation_data = orientation_data.astype(np.float32)
o[cell_data_label + '/Eulers'] = orientation_data.reshape(tuple(dummy)+(3,))

# Attributes to CellData group
o[cell_data_label].attrs['AttributeMatrixType'] = np.array([3],np.uint32)
o[cell_data_label].attrs['TupleDimensions']     = np.array(dummy,np.uint64)
    
# Common Attributes for groups in CellData
for group in ['/Phases','/Eulers']:
  o[cell_data_label + group].attrs['DataArrayVersion']      = np.array([2],np.int32)
  o[cell_data_label + group].attrs['Tuple Axis Dimensions'] = 'x={},y={},z={}'.format(*np.array(dummy))

# phase attributes
o[cell_data_label + '/Phases'].attrs['ComponentDimensions'] = np.array([1],np.uint64)
o[cell_data_label + '/Phases'].attrs['ObjectType']          = 'DataArray<int32_t>'
o[cell_data_label + '/Phases'].attrs['TupleDimensions']     = np.array(dummy,np.uint64)

# Quats attributes
o[cell_data_label + '/Eulers'].attrs['ComponentDimensions'] = np.array([3],np.uint64)
o[cell_data_label + '/Eulers'].attrs['ObjectType']          = 'DataArray<float>'        
o[cell_data_label + '/Eulers'].attrs['TupleDimensions']     = np.array(dummy,np.uint64)

# Create EnsembleAttributeMatrix
ensemble_label = data_container_label + '/EnsembleAttributeMatrix' 

# Data CrystalStructures
o[ensemble_label + '/CrystalStructures'] = np.uint32(np.array([999,1]))
#                                                Crystal_structures[f.get_crystal_structure()]])).reshape((2,1))
o[ensemble_label + '/PhaseTypes']        = np.uint32(np.array([999,Phase_types['Primary']])).reshape((2,1))    # ToDo

# Attributes Ensemble Matrix
o[ensemble_label].attrs['AttributeMatrixType'] = np.array([11],np.uint32)
o[ensemble_label].attrs['TupleDimensions']     = np.array([2], np.uint64)

# Attributes for data in Ensemble matrix
for group in ['CrystalStructures','PhaseTypes']: # 'PhaseName' not required MD: But would be nice to take the phase name mapping
  o[ensemble_label+'/'+group].attrs['ComponentDimensions']   = np.array([1],np.uint64)
  o[ensemble_label+'/'+group].attrs['Tuple Axis Dimensions'] = 'x=2'
  o[ensemble_label+'/'+group].attrs['DataArrayVersion']      = np.array([2],np.int32)
  o[ensemble_label+'/'+group].attrs['ObjectType']            = 'DataArray<uint32_t>'
  o[ensemble_label+'/'+group].attrs['TupleDimensions']       = np.array([2],np.uint64)
    
# Create geometry info
geom_label = data_container_label + '/_SIMPL_GEOMETRY'

o[geom_label + '/DIMENSIONS'] = np.int64(np.array(dummy))
o[geom_label + '/ORIGIN']     = np.float32(np.zeros(3))
o[geom_label + '/SPACING']    = np.float32(np.array(dummy)*4)
    
o[geom_label].attrs['GeometryName']     = 'ImageGeometry'
o[geom_label].attrs['GeometryTypeName'] = 'ImageGeometry'
o[geom_label].attrs['GeometryType']          = np.array([0],np.uint32) 
o[geom_label].attrs['SpatialDimensionality'] = np.array([3],np.uint32) 
o[geom_label].attrs['UnitDimensionality']    = np.array([3],np.uint32) 

