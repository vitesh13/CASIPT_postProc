#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import h5py
import argparse
import math
import numpy as np
from Fe_decomposition import Decompose

def eulers_toR(eulers):
  """
  This function returns a rotation matrix from given euler angle
  The rotation matrix would be the rotational part of the elastic deformation gradient
  
  """
  r_matrix = np.zeros((3,3))
  c1 = math.cos(eulers[0])
  c2 = math.cos(eulers[1])
  c3 = math.cos(eulers[2])
  s1 = math.sin(eulers[0])
  s2 = math.sin(eulers[1])
  s3 = math.sin(eulers[2])

  # copying from function eu2om in rotations.f90
  r_matrix[0][0] =  c1*c3 - s1*c2*s3
  r_matrix[0][1] =  s1*c3 + c1*s3*c2 
  r_matrix[0][2] =  s3*s2

  r_matrix[1][0] = -c1*s3 - s1*c3*c2 
  r_matrix[1][1] = -s1*s3 + c1*c3*c2 
  r_matrix[1][2] =  c3*s2

  r_matrix[2][0] =  s1*s2
  r_matrix[2][1] = -c1*s2
  r_matrix[2][2] =  c2

  r_matrix = r_matrix.transpose()

  return r_matrix

def findFe_initial(F,Fp):
  """
  This function returns elastic deformation gradient from multiplicative decomposition. 
  Assumes, F = F_e F_p.
  
  """
  
  Fe = np.matmul(F,np.linalg.inv(Fp))
  return Fe

def om2eu(om):
  if abs(om[2][2]) < 1.0:
    zeta = 1.0/math.sqrt(1.0-om[2][2]**2.0)
    eu = np.array([math.atan2(om[2][0]*zeta,-om[2][1]*zeta), \
          math.acos(om[2][2]), \
          math.atan2(om[0][2]*zeta, om[1][2]*zeta)])
  else:
    eu = np.array([math.atan2(om[0][1],om[0][0]),0.5*math.pi*(1-om[2][2]),0.0])
  
  eu = np.where(eu<0.0,(eu+2.0*math.pi)%np.array([2.0*math.pi,math.pi,2.0*math.pi]),eu)
  
  return eu

def eu2qu(eu):
  ee = 0.5*eu

  cPhi = math.cos(ee[1])
  sPhi = math.sin(ee[1])
  P = -1.0
  qu =   np.array([   cPhi*math.cos(ee[0]+ee[2]), \
                   -P*sPhi*math.cos(ee[0]-ee[2]), \
                   -P*sPhi*math.sin(ee[0]-ee[2]), \
                   -P*cPhi*math.sin(ee[0]+ee[2])])

  if qu[0] < 0.0:
    qu = qu*(-1.0)

  return qu
 

def qu2om(qu):
  qq = qu[0]**2 - (qu[1]**2 + qu[2]**2 + qu[3]**2)
  om = np.zeros((3,3))
  om[0][0] = qq + 2.0*qu[1]*qu[1]
  om[1][1] = qq + 2.0*qu[2]*qu[2]
  om[2][2] = qq + 2.0*qu[3]*qu[3]

  om[0][1] = 2.0*(qu[1]*qu[2] - qu[0]*qu[3])
  om[1][2] = 2.0*(qu[2]*qu[3] - qu[0]*qu[1])
  om[2][0] = 2.0*(qu[3]*qu[1] - qu[0]*qu[2])
  om[1][0] = 2.0*(qu[2]*qu[1] + qu[0]*qu[3])
  om[2][1] = 2.0*(qu[3]*qu[2] + qu[0]*qu[1])
  om[0][2] = 2.0*(qu[1]*qu[3] + qu[0]*qu[2])

  return om 
 
parser = argparse.ArgumentParser()

# --------------------------------------------------------------------
#                                MAIN
# --------------------------------------------------------------------
parser.add_argument('filenames', nargs='+',
                    help='restart files')
parser.add_argument('--casipt', nargs='+',
                    help='casipt input files')

options = parser.parse_args()

with open(options.casipt[0],'r') as f:
  data = f.readlines()

hdf_file = h5py.File(options.filenames[0],'a') 

rho_CA = np.loadtxt('resMDRX._rho.txt')
for i in range(np.array(hdf_file['/constituent/1_omega_plastic']).shape[1]):
  hdf_file['/constituent/1_omega_plastic'][:,i] = rho_CA/48   # for BCC till 48, but for fcc till 24 only 

for i,line in enumerate(data):
  index = int(line.split()[1])
  hdf_file['/constituent/1_omega_plastic'][index,0:48] = 1E12  # for BCC till 48, but for fcc till 24 only
# in the code F = Fe.Fp 
  Fp = np.array(hdf_file['Fp'][index]).reshape((3,3))
  F  = np.array(hdf_file['F'][index]).reshape((3,3))
  Fe = findFe_initial(F.T,Fp.T) # because restart file stores deformation gradients as transposed form 
  d = Decompose(Fe)
  R = d.math_rotationalPart33(Fe)  #rotational part of Fe = RU
  # to enable gradual change in F
  # -----------------------------
  orig_eulers = om2eu(R.transpose())   #in radians O_m = R.transpose()
  #orig_eulers = orig_eulers*math.pi/180.0
  print('original euler:',orig_eulers,'index:',index) 
  # -----------------------------
  stretch = np.matmul(np.linalg.inv(R),Fe)
  eulers = np.array([float(line.split()[5]),float(line.split()[7]),float(line.split()[9])])
  eulers = eulers*math.pi/180.0 #degrees to radians
  print('euler aim:',eulers)
  # -----------------------------
  #diff_eulers = eulers - orig_eulers
  #eulers = orig_eulers + diff_eulers*1.0 #gradual change in 4 steps TODO: need to check how many steps are required for stability
  #print('gradual euler:',eulers)
  # -----------------------------
  rotation_new = eulers_toR(eulers) #you get rotation matrix R from this function 
  Fe_new       = np.matmul(rotation_new,stretch)
  Fp_new       = np.matmul(F,np.linalg.inv(Fe_new))
  Fp_new       = Fp_new.T           # because restart file stores deformation gradients as transposed form
  hdf_file['Fp'][index] = Fp_new.reshape((1,1,3,3))
  #------SANITY CHECK---------------------------------------



 
