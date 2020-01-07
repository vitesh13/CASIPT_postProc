#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np
import math
import h5py

#replicate everything that is in DAMASK
#then we try to cut down the things to get it more pythonic

class Decompose():
# ------------------------------------------------------------------
  def __init__(self,
              m):   #m is the matrix to decompose
    self.m = m

  def math_detSym33(self,m):
    math_detSym33 = -(m[0][0]*m[1][2]**2.0 + m[1][1]*m[0][2]**2.0 + m[2][2]*m[0][1]**2.0) \
                    + m[0][0]*m[1][1]*m[2][2] + 2.0*m[0][1]*m[0][2]*m[1][2]
    return math_detSym33

  def math_invariantsSym33(self,m):
    f_math_invariantsSym33 = np.zeros(3)
    f_math_invariantsSym33[0] = np.trace(m)
    f_math_invariantsSym33[1] = m[0][0]*m[1][1] + m[0][0]*m[2][2] + m[1][1]*m[2][2] \
                            -(m[0][1]**2      + m[0][2]**2      + m[1][2]**2)
    f_math_invariantsSym33[2] = self.math_detSym33(m) #np.linalg.det(m)

    return f_math_invariantsSym33  


  def math_eigenvectorBasisSym33(self,m):
    tol = 1E-14
    invariants = self.math_invariantsSym33(m)
    EB = np.zeros((3,3,3))
    N  = np.zeros((3,3,3))
    
    P = invariants[1]-invariants[0]**2.0/3.0
    Q = -(2.0/27.0)*invariants[0]**3.0 + np.product(invariants[0:2])/3.0-invariants[2]
    
    values = np.zeros(3)
    if np.all(abs(np.array([P,Q]))<1E-14):
      values[:] = invariants[0]/3.0
      EB[0][0][0] = 1.0
      EB[1][1][1] = 1.0
      EB[2][2][2] = 1.0
    else:
      rho = math.sqrt(-3.0*P**3.0)/9.0  #if this gives error then we need to somehow allow complex numbers?? check fortran code
      phi = math.acos(-Q/rho*0.5)
      if phi > 1.0: 
        phi = 1.0
      elif phi < -1.0:
        phi = -1.0

      values = 2.0*rho**(1.0/3.0)* \
                              np.array([math.cos(phi/3.0), \
                               math.cos((phi+2.0*math.pi)/3.0), \
                               math.cos((phi+4.0*math.pi)/3.0) \
                              ]) + invariants[0]/3.0
      N[:][:][0] = m - values[0]*np.identity(3)
      N[:][:][1] = m - values[1]*np.identity(3)
      N[:][:][2] = m - values[2]*np.identity(3)  #Only diagonal elements seem to change, so should be fine

      if abs(values[0] - values[1]) == 0.0:
        EB[:][:][2] = np.matmul(N[:][:][0],N[:][:][1]/ \
                                           ((values[2]-values[0])*(values[2]-values[1])))
        EB[:][:][0] = np.identity(3) - EB[:][:][2]
      elif abs(values[1] - values[2]) == 0.0:  
        EB[:][:][0] = np.matmul(N[:][:][1],N[:][:][2]/ \
                                           ((values[0]-values[1])*(values[0]-values[2])))
        EB[:][:][1] = np.identity(3) - EB[:][:][0]
      elif abs(values[2] - values[0]) == 0.0:  
        EB[:][:][1] = np.matmul(N[:][:][0],N[:][:][2]/ \
                                           ((values[1]-values[0])*(values[1]-values[2])))
        EB[:][:][0] = np.identity(3) - EB[:][:][1]
      else:
        EB[:][:][0] = np.matmul(N[:][:][1],N[:][:][2]/ \
                                           ((values[0]-values[1])*(values[0]-values[2])))
        EB[:][:][1] = np.matmul(N[:][:][0],N[:][:][2]/ \
                                           ((values[1]-values[0])*(values[1]-values[2])))
        EB[:][:][2] = np.matmul(N[:][:][0],N[:][:][1]/ \
                                           ((values[2]-values[0])*(values[2]-values[1])))

    math_eigenvectorBasisSym33 = math.sqrt(values[0])*EB[:][:][0] + math.sqrt(values[1])*EB[:][:][1] + math.sqrt(values[2])*EB[:][:][2] 

    return math_eigenvectorBasisSym33 

  def math_rotationalPart33(self,m):

    U = self.math_eigenvectorBasisSym33(np.matmul(np.transpose(m),m))  # this diff due to row and columnn major diff in python and Fortran

    Uinv = np.linalg.inv(U)
    
    if np.all(Uinv<=0.0):
      math_rotationalPart33 = np.identity(3)
    else:
      math_rotationalPart33 = np.matmul(m,Uinv)    # this diff due to row and columnn major diff in python and Fortran 

    return math_rotationalPart33 

  
  
