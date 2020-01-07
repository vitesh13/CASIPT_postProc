#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import h5py
import argparse
import math
import numpy as np
from Fe_decomposition import Decompose

parser = argparse.ArgumentParser()
# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------
parser.add_argument('filenames', nargs='+',
                    help='restart files')
options = parser.parse_args()
# --------------------------------------------------------------------
class Ori_creator():
  
  def __init__(self,hdf): #hdf is the name of the hdf5 file to read
    self.hdf = h5py.File(hdf,'r')

  def om2eu(self,om):
    if abs(om[2][2]) < 1.0:
      zeta = 1.0/math.sqrt(1.0-om[2][2]**2.0)
      eu = np.array([math.atan2(om[2][0]*zeta,-om[2][1]*zeta), \
            math.acos(om[2][2]), \
            math.atan2(om[0][2]*zeta, om[1][2]*zeta)])
    else:
      eu = np.array([math.atan2(om[0][1],om[0][0]),0.5*math.pi*(1-om[2][2]),0.0])
    
    eu = np.where(eu<0.0,(eu+2.0*math.pi)%np.array([2.0*math.pi,math.pi,2.0*math.pi]),eu)
    
    return eu
   
  def get_F(self):
    self.F = np.array(self.hdf['convergedF'])
    self.F = np.reshape(self.F,(len(self.F),3,3))

  def get_Fp(self):
    self.Fp = np.array(self.hdf['convergedFp'])
    self.Fp = np.reshape(self.Fp,(len(self.Fp),3,3))
 
  def findFe_initial(self,Fp,F):
    
    Fe = np.matmul(F,np.linalg.inv(Fp))
    return Fe
 

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
o = Ori_creator(options.filenames[0])
o.get_Fp()  
o.get_F()
#F_total = o.F
#Fp_total = o.Fp
orientation_array       = np.zeros((len(o.F),3))

for i in range(len(o.F)):
  Fe = o.findFe_initial(o.Fp[i].T,o.F[i].T)  #transpose needed because the restart files stored the F as transpose
  d = Decompose(Fe)
  R = d.math_rotationalPart33(Fe)  #rotational part of Fe = RU
  if i == 0:
    print(R) 
  orientation_array[i] = o.om2eu(R.T)
  if i == 0:
    print(orientation_array) 


grid = []
with open('resMDRX.3D.geom','r') as f:
  for i, line in enumerate(f):
    if i < 1:
      grid.append(line)

dummy = [grid[0].split()[2], grid[0].split()[4], grid[0].split()[6]]
dummy = [int(i) for i in dummy]

orientation_data = orientation_array
print('orientation_data is:', orientation_data)

#--------------------------------------------------------------------------
o = h5py.File('new_restart_geom.dream3D','w')
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



