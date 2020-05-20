#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import os,sys,math,re,time,struct
import h5py
import re
import numpy as np
import os, sys, shutil
import scipy
from scipy import spatial
import argparse
import subprocess
import damask
import pandas as pd
import itertools as it
# import matplotlib.pyplot as plt
# plt.switch_backend('tkagg')             # this only for plotting at the moment, delete it later.

from scipy.linalg import polar
from numpy.linalg import inv
import shutil
from imp import reload

#
# codes_dir='/nethome/storage/raid4/k.sedighiani/0_git/FittingRoutine/Remeshing/'
# sys.path.append(codes_dir)

# import modules as my
#from damask import rgposth5 as p5
#from damask import regriding as rgg
from damask import regriding as rgg




# directory
main_dir = os.getcwd()

# ------------------------------------------------------------------------------
#                                reading the input files
# ------------------------------------------------------------------------------
description = """
Re-gridding:
It is a tool to replace a severely deformed grid with a new undeformed grid.
In this method, the solution from the deformed grid is mapped on the new undeformed grid and the simulation is restarted.

Generated file for restart analysis:
    - a new geom file with updated microstructure
    - a new output (hdf5) file for restart analysis

Note: The original geom and output files will be kept untouched.
To restart the simulation, the generated files need to be renamed properly.

Defaults for the newly generated grid:
    - The default for the re-gridding is to keep the smallest deformed grid size constant.
      This increases in the simulation resolution.
      Note: The default can be overwritten using the option: "scale"
    - The last converged increment (requested) is used to build the re-gridding files.

"""

parser = argparse.ArgumentParser(description = description)

# required
# ----------------------------------------------------------------------------
myhelp =    """(required) the original geom file name."""

parser.add_argument('-g','--geom', dest = 'geom', metavar = 'string',
                    help = myhelp)
# ----------------------------------------------------------------------------
myhelp =    """(required) the original load file name."""

parser.add_argument('-l','--load', dest = 'load', metavar = 'string',
                    help = myhelp)
# optional
# ----------------------------------------------------------------------------
myhelp = """
(optional) \
defualt: 1.0


A scale for the default new grid size.
refernce resolution: the new grid is estimated based on keeping the smallest deformed grid size constant.
Default: using the reference resolution (scale = 1.0)

options:
        ** one float: used as a scale to increase or decrease the resolution.
                    e.g. 2 for increasing the resolution to 2 times of the refrence resolution.
Note: using "grid" the new grid can be strictly determine by user.
"""

parser.add_argument('-s','--scale', dest = 'scale', type = float, nargs='*', metavar = 'float', default=1.0,
                    help = myhelp)
# ----------------------------------------------------------------------------
myhelp = """
(optional) \
defualt: False

User defined new grid.
    ** 3 integers: strictly determine the number of new grid in x, y, z directions
"""

parser.add_argument('-n','--grid', dest = 'grid', type = int, nargs='*', metavar = 'int', default=False,
                    help = myhelp)
# ----------------------------------------------------------------------------
myhelp = """
(optional) \
Default: False

To plot re-grided data.
"""

parser.add_argument('-p','--plot', dest = 'plot', type=bool, metavar = 'Boolean', default=False,
                    help = myhelp)
# ----------------------------------------------------------------------------
myhelp = """
(optional) \
defualt: False

The input is 6 float numbers [x_lower_ratio, x_upper_ratio, y_lower_ratio, y_upper_ratio, z_lower_ratio, z_upper_ratio,]
which shows the coordinates of the submodel as a ratio of the original RVE size.
"""

parser.add_argument('-m','--submodel', dest = 'submodel', type = float, nargs='*', metavar='float',
                    default=False, help = myhelp)

# ----------------------------------------------------------------------------
myhelp = """
(optional) \
defualt: "full"

reset the deformation tensor:
        "elastic:" keep the elastic deformation
        "full": reset the deformation tensor to identity
"""

parser.add_argument('-r','--reset', dest = 'reset',  metavar='string',
                    default='full', help = myhelp)
# ----------------------------------------------------------------------------
# set inputs

options = parser.parse_args()

# options.geom = '62G.geom'
# options.load = 'CompX.load'


geom_name = p5.file_util.remove_fileFormat(options.geom)
load_name = p5.file_util.remove_fileFormat(options.load)
reset = p5.setting_util.set_reset(options.reset)
scale = p5.setting_util.set_scale(options.scale, options.grid)
sub_coefs = p5.setting_util.set_submodel(options.submodel)

# default for material config and hdf5 restart files
mat = 'material.config'
hdf5_name = '%s_%s.hdf5'%(geom_name,load_name)
history_name = '%s_%s_regriddingHistory.hdf5'%(geom_name,load_name)




# -------------------------------------------------------------------------------------------------------------------------------
# *****
# -------------------------------------------------------------------------------------------------------------------------------




inc= 'last'

# # inc = 800


# regriding for the inc
rg = rgg.geom_regridder(geom_name,load_name)
rg.set_dir(main_dir)
rg.set_plotting(options.plot)
rg.set_RVEscale(scale)
grid0, size0, origin0, microstructure0_flatten, geom_comments0 = rg.read_geom()
print(origin0)
df_cell, df_nodal = rg.make_displacements_df (inc = inc)
rebacked_id, new_grid = rg.report_restartGeom(df_cell, df_nodal, sub_coefs)
# # Cauchy and F
# # rgh5 = rgg.regridding_history_maker(geom_name, load_name)
# # rgh5.set_dir(main_dir)
# # rgh5.read_h5F(inc = inc)
# # rgh5.read_historyOut()
# # rgh5.build_regriddingHistory_dict(rebacked_id)
# # F_rg_hist = rgh5.history_dict[str(rgh5.inc)]['F']
# # Cauchy_strain_hist = rgh5.history_dict[str(rgh5.inc)]['Cauchy_strain']
# #------------------------------------------
# # Caucy strain from accomulated F
# # Cauchy_strain_accomF = np.array(p5.outProc_util.make_left_Cauchy_strain(F_rg_hist))
# # Cauchy_strain_accomF = Cauchy_strain_accomF[rebacked_id]




# Building the new coordinates
elem0 = int(grid0.prod())
elem_rg = int(new_grid.prod())

New_RVE_size = rg.New_RVE_size
new_grid_cell = rg.new_grid
origin0 = rg.origin0
Cell_coords = p5.math_util.create_grid(new_grid_cell, New_RVE_size, origin0, point_type='Cell')
#------------------------------------------
# reading main inputs for processing
out5 = p5.output_reader(hdf5_name)
inc, inc_key = out5.make_incerement(inc)
phase_name = out5.constituents[0]
# reading main inputs
orientations = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['orientation'])
orientations = np.array(orientations.tolist())
# F = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['F'])
# Fp = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['Fp'])
# Pio = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['P'])
# #------------------------------------------
## grain rotation
grain_rotation = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['grain_rotation'])  #need to add [:,3] is using damask generated output
# if generated by python Result class, then remove it
grain_rotation_rg = grain_rotation[rebacked_id]
grain_rotation_rg_scalar = grain_rotation_rg
# euler angles
ou = p5.outProc_util()
eulers = np.array(ou.make_eulersFromQuaternion(orientations))
eulers_rg = eulers[rebacked_id]
# dislocation density
rho_m = np.array(out5.data[inc_key]['constituent'][phase_name]['plastic']['rho_mob'])
rho_d = np.array(out5.data[inc_key]['constituent'][phase_name]['plastic']['rho_dip'])
rho = rho_m + rho_d
# reshaped
rho_m_rg = rho_m[rebacked_id]
rho_d_rg = rho_d[rebacked_id]
rho_rg = rho[rebacked_id]
rho= np.sum(rho_rg,axis=1)
# subgrain sizes
r_s   = np.array(out5.data[inc_key]['constituent'][phase_name]['plastic']['r_s']) 
r_s_rg   = r_s[rebacked_id]

print('orientation shape before: ',np.shape(orientations))
orientations_rg = orientations[rebacked_id] #hopefully it works
print('orientation shape after: ',np.shape(orientations_rg))


#------------------------------------------
# current Cauchy strain
# Cauchy_strain = np.array(p5.outProc_util.make_left_Cauchy_strain(F))
# Cauchy_strain = Cauchy_strain[rebacked_id]

#------------------------------------------
# cauchy stress (which F?)
# cauchy_stress = np.array(ou.make_Cauchy_stress(F, Pio))
# cauchy_stress = cauchy_stress[rebacked_id]
# cauchy_stress =cauchy_stress.reshape(cauchy_stress.shape[0],3,3)

#------------------------------------------
# Mises stress
# Mises_stress_hist, Mises_strain_hist = p5.outProc_util.Mises_point(cauchy_stress, Cauchy_strain_hist)
# Mises_stress_accom, Mises_strain_accom = p5.outProc_util.Mises_point(cauchy_stress, Cauchy_strain_accomF)
# Mises_stress, Mises_strain = p5.outProc_util.Mises_point(cauchy_stress, Cauchy_strain)


# Trs_eps = p5.outProc_util.Principal_shear(Cauchy_strain_hist)
# Trs_S = p5.outProc_util.Principal_shear(cauchy_stress)/2.0


#------------------------------------------
# make output table file
#------------------------------------------

#--------------
# make df
#---------------

# # some information
df = pd.DataFrame() #{'inc': inc*np.ones(elem_rg, dtype=int)}
#df['elem'] = range(1,elem_rg+1)
# coords for new grids
Cell_coords = Cell_coords*600E-06
df['x'] = Cell_coords[:,0]
df['y'] = Cell_coords[:,1]
df['z'] = Cell_coords[:,2]
# # initial grain
df['grain'] = np.array(df_cell['grain'])[rebacked_id]
## Rotation
df['Rotation'] = grain_rotation_rg_scalar
# # total dislo
df['rho'] = rho
# # subgrain sizes 
df['r_s'] = r_s_rg
# euler angles
df['phi1'] = eulers_rg[:,0]
df['PHI'] = eulers_rg[:,1]
df['phi2'] = eulers_rg[:,2]
#
# df['Phases'] = np.ones(df.shape[0])

#print(max(df['grain']))
# # Mises strains
# df['Meps_h'] = Mises_strain_hist
# df['Meps_a'] = Mises_strain_accom
# df['Meps'] = Mises_strain
# # Mises stress
# df['Msig_h'] = Mises_stress_hist
# df['Msig_a'] = Mises_stress_accom
# df['Msig'] = Mises_stress
# df['Trs_eps'] = Trs_eps
# df['Trs_S'] = Trs_S


# header
header_f = '%s %s\n'%(str(int(np.prod(new_grid))),str(max(df['grain']))) #,str(max(df['grain']))

output = '%s_%s'%(geom_name,load_name)
file_rg = os.path.join(main_dir,'postProc','%s_%s.txt'%(output,inc))
if not os.path.exists(os.path.join(main_dir,'postProc')):
    os.makedirs(os.path.join(main_dir,'postProc'))

with open(file_rg,'w') as f:
    f.write(header_f)
    df.to_string(f,header=False,formatters=["{:.8f}".format,"{:.8f}".format,"{:.8f}".format, \
                                            "{:.8f}".format,"{:.8f}".format,"{:.6E}".format, \
                                            "{:.12f}".format,"{:.8f}".format,"{:.8f}".format, \
                                            "{:.8f}".format],index=False)

# create a HDF5 file to creat IPF of 3D data from DREAM3D : so modify the existing hdf5 output file
# added by Vitesh Shah for visualization purposes
# to be removed once the addIPFcolor on HDF5 files works
# dir_src = os.getcwd()
# shutil.copy(dir_src + '/' + hdf5_name,dir_src + '/' + 'new_' + geom_name + '_' + load_name + '.hdf5') #copying and renaming the file
new_hdf_name = 'new_' + geom_name + '_' + load_name + '.hdf5'
hdf = h5py.File(new_hdf_name,'w')
hdf.attrs['DADF5_version_major'] = 0
hdf.attrs['DADF5_version_minor'] = 6
hdf.attrs['DADF5-version'] = 0.2
hdf.create_group('geometry')
hdf['geometry'].attrs['grid'] = np.array(rg.new_grid, np.int32)
hdf['geometry'].attrs['size'] = np.array(rg.New_RVE_size, np.float64)
hdf['geometry'].attrs['origin'] = np.array(rg.origin0, np.float64)

#mapping data
comp_dtype = np.dtype([('Name',np.string_,64),('Position',np.int32)])
new_len    = np.prod(np.int32(rg.new_grid))
data_name  = [phase_name]*int(new_len)
data_value = [i for i in range(new_len)]
new_data   = list(zip(data_name,data_value))
new_data   = np.array(new_data,dtype=comp_dtype)
new_data   = new_data.reshape(new_len,1)
dataset    = hdf.create_dataset("/mapping/cellResults/constituent",(new_len,1),comp_dtype)
dataset[...] = new_data

data_name  = ['1_SX']*int(new_len)
new_data   = list(zip(data_name,data_value))
new_data   = np.array(new_data,dtype=comp_dtype)
new_data   = new_data.reshape(new_len,1)
dataset    = hdf.create_dataset("/mapping/cellResults/materialpoint",(new_len,1),comp_dtype)
dataset[...] = new_data

#orientation_rg
comp_dtype  = np.dtype([('w',np.float64),('x',np.float64),('y',np.float64),('z',np.float64)])
new_len     = np.prod(np.int32(rg.new_grid))
dataset_ori = hdf.create_dataset("/{}/constituent/{}/generic/orientation".format(inc_key,phase_name),(new_len,),comp_dtype)
orientations_rg = np.array([tuple(i) for i in orientations_rg[:]],dtype=comp_dtype)
dataset_ori[...] = orientations_rg
hdf["/{}/constituent/{}/generic/orientation".format(inc_key,phase_name)].attrs['Lattice'] = 'bcc'

#rho_rg and grain_rotation_rg_scalar
dataset_rho = hdf.create_dataset("/{}/constituent/{}/plastic/tot_density".format(inc_key,phase_name),(new_len,))
dataset_rho[...] = rho

dataset_rot = hdf.create_dataset("/{}/constituent/{}/generic/grain_rotation".format(inc_key,phase_name),(new_len,))
dataset_rot[...] = grain_rotation_rg_scalar

hdf.create_group('/{}/materialpoint/1_SX/generic'.format(inc_key))
hdf.create_group('/{}/materialpoint/1_SX/plastic'.format(inc_key))


#hdf.create_group('inc{}'.format(inc))
print(inc_key)
# dataset_new = hdf[inc_key]['constituent'][phase_name]['generic']['orientation']
# dataset_new[...] = orientations_rg  # refer https://stackoverflow.com/questions/22922584/how-to-overwrite-array-inside-h5-file-using-h5py for details
    
## calculation to find the number of microstructures/texture







































#
#
#
#
# type = 'regrided_out' # deformed_out
#
#
#
# # history
# hist5 = p5.history_reader(history_name)
# out5 = p5.output_reader(hdf5_name)
#
# if hist5.incs[-1] == out5.incs[-1]:
#     hist5.active_inc = hist5.incs[-1]
#
# # elif int(hist5.incs[-1]) < int(out5.incs[0]):
# #     hist5.active_inc = hist5.incs[-1]
# # else:
# #     hist5.active_inc = hist5.incs[-2] # not sure, check it
#
#
# inci, inc_key = out5.make_incerement(inci)
# phase_name = out5.constituents[0]
#
#
#
#
# rg = rgg.geom_regridder(geom_name,load_name)
# rg.set_dir(main_dir)
# rg.set_plotting(options.plot)
# rg.set_RVEscale(scale)
# grid0, size0, origin0, microstructure0_flatten, geom_comments0 = rg.read_geom()
# df_cell, df_nodal = rg.make_displacements_df ()
# if inci == int(hist5.incs[-1]):
#     rebacked_id = hist5.data[hist5.active_inc]['id']
#     F_rg_hist = hist5.data[hist5.active_inc]['F']
#     Cauchy_strain_hist = hist5.data[hist5.active_inc]['Cauchy_strain']
#     ave_cau = np.einsum('ijk->jk',Cauchy_strain_hist)/len(Cauchy_strain_hist)
#     rebacked_id, new_grid = rg.report_restartGeom(df_cell, df_nodal, sub_coefs)
# else:
#     rebacked_id, new_grid = rg.report_restartGeom(df_cell, df_nodal, sub_coefs)
#     rgh5 = rgg.regridding_history_maker(geom_name, load_name)
#     rgh5.set_dir(main_dir)
#     rgh5.read_h5F()
#     rgh5.read_historyOut()
#     rgh5.build_regriddingHistory_dict(rebacked_id)
#     rgh5.build_regriddingHistory()
#     #
#     F_rg_hist = rg.history_dict[str(inci)]['F']
#     Cauchy_strain_hist = rg.history_dict[str(inci)]['Cauchy_strain']
#
#
#
#
# elem0 = int(grid0.prod())
# elem_rg = Cauchy_strain_hist.shape[0]
#
#
# orientations = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['orientation'])
# orientations = np.array(orientations.tolist())
# F = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['F'])
# Fp = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['Fp'])
# Pio = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['P'])
# #
# #
# #
# ou = p5.outProc_util()
# eulers = np.array(ou.make_eulersFromQuaternion(orientations))
# eulers_rg = eulers[hist5.data[hist5.incs[-1]]['id']]
# #
# #
# New_RVE_size = rg.New_RVE_size
# new_grid_cell = rg.new_grid
# origin0 = rg.origin0
# Cell_coords = p5.math_util.create_grid(new_grid_cell, New_RVE_size, origin0, point_type='Cell')
#
#
#
# rho_m = np.array(out5.data[inc_key]['constituent'][phase_name]['plastic']['rho_mob'])
# rho_d = np.array(out5.data[inc_key]['constituent'][phase_name]['plastic']['rho_dip'])
# rho = rho_m + rho_d
#
# rho_m_rg = rho_m[rebacked_id]
# rho_d_rg = rho_d[rebacked_id]
# rho_rg = rho[rebacked_id]
#
# rho= np.sum(rho_rg,axis=1)
#
#
# Cauchy_strain = np.array(p5.outProc_util.make_left_Cauchy_strain(F_rg_hist))
# Cauchy_strain = Cauchy_strain.reshape(Cauchy_strain.shape[0],3,3)
# Cauchy_strain = Cauchy_strain[rebacked_id]
#
#
# strain_ids = ['e_%d'% i for i in range(1,10) ]
# stress_ids = ['S_%d'% i for i in range(1,10) ]
#
#
#
# cauchy_stress = np.array(ou.make_Cauchy_stress(F, Pio))
# cauchy_stress = cauchy_stress[rebacked_id]
# cauchy_stress =cauchy_stress.reshape(cauchy_stress.shape[0],3,3)
#
#
# Mises_stress, Mises_strain = p5.outProc_util.Mises_point(cauchy_stress, Cauchy_strain_hist)
# Mises_stress1, Mises_strain1 = p5.outProc_util.Mises_point(cauchy_stress, Cauchy_strain)
#
# ####################################################################
# df = pd.DataFrame({'inc': inci*np.ones(elem_rg, dtype=int)})
#
# df['elem'] = range(1,elem_rg+1)
# df['x'] = Cell_coords[:,0]
# df['y'] = Cell_coords[:,1]
# df['z'] = Cell_coords[:,2]
# #
# df['phi1'] = eulers_rg[:,0]
# df['PHI'] = eulers_rg[:,1]
# df['phi2'] = eulers_rg[:,2]
# df['Phases'] = np.ones(df.shape[0])
#
# df['MS'] = Mises_strain
# df['MS1'] = Mises_strain1
#
# df['MSS'] = Mises_stress
# df['MSS1'] = Mises_stress1
# df['rho'] = rho
#
# output = '%s_%s'%(geom_name,load_name)
# file_rg = os.path.join(main_dir,'postProc','DAMASK_CA_%s_%s.txt'%(output,inci))
# if not os.path.exists(os.path.join(main_dir,'postProc')):
#     os.makedirs(os.path.join(main_dir,'postProc'))
#
# with open(file_rg,'w') as f:
#     df.to_string(f,index=False)
#
#
# #
#
#
#
#
#
#
#
#
# for i in range(9):
#     col_title = '%s_ln(V)'%(i+1)
#     df[col_title] = Cauchy_strain[:,i]
#
#
#
#
#
#
#
# if type == 'regrided_out':
# elif type == 'deformed_out':
#     a=3
#
#
#
#
# F1 = hist5.data['1000']['F']
# F2 = hist5.data['2000']['F']
# F_2 = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['F'])
# Fp_2 = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['Fp'])
#
#
# vecDot = np.vectorize(np.dot, signature='(m,p),(p,n)->(m,n)')
#
#
# r1,v1,u1 = p5.math_util.RVU_decomposition(F1, ['V', 'U'])
# r2,v2,u2 = p5.math_util.RVU_decomposition(F2, ['V', 'U'])
# r_2,v_2,u_2 = p5.math_util.RVU_decomposition(F_2, ['V', 'U'])
# rp_2,vp_2,up_2 = p5.math_util.RVU_decomposition(Fp_2, ['V', 'U'])
#
#
# F = vecDot(u_2, u1)
#
#
# F_rg_hist =vecDot(up_2, F1)
#
#
# Cauchy_strain = np.array(p5.outProc_util.make_left_Cauchy_strain(F_rg_hist))
# Cauchy_strain = Cauchy_strain.reshape(Cauchy_strain.shape[0],3,3)
# Cauchy_strain = Cauchy_strain[rebacked_id]
#
#
#
#
# Cauchy_strain2 = np.array(p5.outProc_util.make_left_Cauchy_strain(F))
# Cauchy_strain2 = Cauchy_strain2.reshape(Cauchy_strain2.shape[0],3,3)
# Cauchy_strain2 = Cauchy_strain2[rebacked_id]
#
# Cauchy_strain2 = np.array(p5.outProc_util.make_left_Cauchy_strain(F))
# Cauchy_strain2 = Cauchy_strain2.reshape(Cauchy_strain2.shape[0],3,3)
# Cauchy_strain2 = Cauchy_strain2[rebacked_id]
#
#
# np.isclose(Cauchy_strain2[0], Cauchy_strain_hist[0])
# np.array_str(Cauchy_strain_hist, precision=2)
#
#
#
#
#
#
# ####################################################################
# df = pd.DataFrame({'inc': inci*np.ones(elem_rg, dtype=int)})
#
# df['elem'] = range(1,elem_rg+1)
# df['1_x'] = Cell_coords[:,0]
# df['2_x'] = Cell_coords[:,1]
# df['3_x'] = Cell_coords[:,2]
# #
# df['1_eu'] = eulers_rg[:,0]
# df['2_eu'] = eulers_rg[:,1]
# df['3_eu'] = eulers_rg[:,2]
#
#
#
# for i in range(9):
#     col_title = '%s_ln(V)'%(i+1)
#     df[col_title] = Cauchy_strain_hist.reshape(Cauchy_strain_hist.shape[0],9)[:,i]
#
#
# # add cauchy stress
# new_Pio = []
# for i, item in enumerate(rebacked_id):
#     new_Pio.append(Pio[item])
#
# new_Pio = np.array(new_Pio)
#
#
# for i in range(9):
#     col_title = '%s_Cauchy'%(i+1)
#     df[col_title] = cauchy_stress[:,i]
#     # add mises
#
#
# mises = np.array(ou.Mises_calc_point(df))
# df['Mises_strain'] = mises[:,1]
# df['Mises_stress'] = mises[:,2]
#
# df['Phases'] = np.ones(df.shape[0])
# df = df[['inc', 'elem', '1_x', '2_x', '3_x', '1_eu', '2_eu', '3_eu','Phases', 'Mises_strain','Mises_stress']]
# df.columns = ['inc', 'elem', 'x', 'y', 'z', 'phi1', 'PHI', 'phi2','Phases', 'Mises_strain','Mises_stress']
# #
# output = '%s_%s'%(geom_name,load_name)
# file_rg = os.path.join(main_dir,'postProc','DAMASK_CA_%s_%s.txt'%(output,inci))
# if not os.path.exists(os.path.join(main_dir,'postProc')):
#     os.makedirs(os.path.join(main_dir,'postProc'))
# with open(file_rg,'w') as f:
#     df.to_string(f,index=False)
#
#
#
#
#
#
#
# # 11 551
# # main_dir = main_dir
# ###################################
#
#
#
#
# if type == 'DAMASK_CA':
#     # intial geom data
#     rg = rgg.geom_regridder(geom_name,load_name)
#     rg.plotting = False
#     rg.RVE_grid_scale = 1.0      # default value. make it input
#     grid0, size0, origin0, microstructure0_flatten, geom_comments0 = rg.read_geom()
#     myH5 = p5.DADF5(hdf5_name)
#     df,df_nodal = myH5.make_displacements_df (inc=inci)
#     rebacked_id, new_grid = rg.report_restartGeom(df, df_nodal,sub_coefs,CA=inci)
#     # rebacked_id, new_grid = rg.report_restartGeom(df_cell, df_nodal, main_dir, sub_coefs)
#     # file to open
#     hdf5_file_name = '%s_%s'%(geom_name,load_name)
#     file = os.path.join(main_dir, hdf5_file_name+'.hdf5')
#     f = h5py.File(file,'r')
#     #
#     inc_key = 'inc{:05}'.format(inci)
#     phase_name = myH5.constituents[0]
#     orientations = np.array(f[inc_key]['constituent'][phase_name]['generic']['orientation'])
#     orientations = np.array(orientations.tolist())
#     #
#     F_def = np.array(f[inc_key]['constituent'][phase_name]['generic']['F'])
#     F_def = F_def.reshape((F_def.shape[0],9))
#     #
#     Pio = np.array(f[inc_key]['constituent'][phase_name]['generic']['P'])
#     Pio = Pio.reshape((Pio.shape[0],9))
#     #
#     elem_rg  = int(new_grid.prod())
#     New_RVE_size = rg.New_RVE_size
#     #
#     #
#     #
#     #
#     ou = p5.outProc_util()
#     eulers = np.array(ou.make_eulersFromQuaternion(orientations))
#     new_eulers = []
#     for i, item in enumerate(rebacked_id):
#         new_eulers.append(eulers[item])
#     new_eulers = np.array(new_eulers)
#     #
#     Cell_coords=rg.create_grid(new_grid,New_RVE_size,origin0, point_type='Cell')
#     df = pd.DataFrame({'inc': inci*np.ones(elem_rg, dtype=int)})
#     df['elem'] = range(1,elem_rg+1)
#     df['1_x'] = Cell_coords[:,0]
#     df['2_x'] = Cell_coords[:,1]
#     df['3_x'] = Cell_coords[:,2]
#     #
#     df['1_eu'] = new_eulers[:,0]
#     df['2_eu'] = new_eulers[:,1]
#     df['3_eu'] = new_eulers[:,2]
#     #
#     rho_m = np.array(f[inc_key]['constituent'][phase_name]['plastic']['rho_mob'])
#     rho_d = np.array(f[inc_key]['constituent'][phase_name]['plastic']['rho_dip'])
#     rho = rho_m + rho_d
#     #
#     rho_m_new = []
#     rho_d_new = []
#     rho_new = []
#     for i, item in enumerate(rebacked_id):
#         rho_m_new.append(rho_m[item])
#         rho_d_new.append(rho_d[item])
#         rho_new.append(rho[item])
#     rho_m_new = np.array(rho_m_new)
#     rho_d_new = np.array(rho_d_new)
#     rho_new = np.array(rho_new)
#     #
#     #
#     for i in range(rho_m.shape[1]):
#         col_title = '%s_rho_m'%(i+1)
#         df[col_title] = rho_m_new[:,i]
#     for i in range(rho_m.shape[1]):
#         col_title = '%s_rho_d'%(i+1)
#         df[col_title] = rho_d_new[:,i]
#     for i in range(rho_m.shape[1]):
#         col_title = '%s_rho'%(i+1)
#         df[col_title] = rho_new[:,i]
#     mobile_list = [i for i in df.columns if i.split('_')[-2:]==['rho', 'm']]
#     df['rho_m_point'] = df[mobile_list].sum(axis=1)
#     dipole_list = [i for i in df.columns if i.split('_')[-2:]==['rho', 'd']]
#     df['rho_d_point'] = df[dipole_list].sum(axis=1)
#     dislo_list = [i for i in df.columns if i.split('_')[-1:]==['rho']]
#     df['rho_point'] = df[dislo_list].sum(axis=1)
#     #
#     #   ---- add a few parameter ----
#     #
#     # add left cauchy strain
#     #
#     F_def_new = []
#     for i, item in enumerate(rebacked_id):
#         F_def_new.append(F_def[item])
#     F_def_new = np.array(F_def_new)
#     #
#     #
#     hdf5_comulative_F = '%s_%s_regriddingHistory'%(geom_name,load_name)
#     com_F_file = os.path.join(main_dir, hdf5_comulative_F+'.hdf5')
#     f_F = h5py.File(com_F_file,'r')
#     F_com = np.array(f_F[str(inci)]['F'])
#     #
#     Cauchy_strain = np.array(ou.make_left_Cauchy_strain(F_com))
#     for i in range(9):
#         col_title = '%s_ln(V)'%(i+1)
#         df[col_title] = Cauchy_strain[:,i]
#     # add cauchy stress
#     new_Pio = []
#     for i, item in enumerate(rebacked_id):
#         new_Pio.append(Pio[item])
#     new_Pio = np.array(new_Pio)
#     # cauchy_stress = np.array(ou.make_Cauchy_stress(F_com, new_Pio))
#     cauchy_stress = np.array(ou.make_Cauchy_stress(F_com, new_Pio))
#     for i in range(9):
#         col_title = '%s_Cauchy'%(i+1)
#         df[col_title] = cauchy_stress[:,i]
#     # add mises
#     mises = np.array(ou.Mises_calc_point(df))
#     df['Mises_strain'] = mises[:,1]
#     df['Mises_stress'] = mises[:,2]
#     #
#     #
#     #
#     df['Phases'] = np.ones(df.shape[0])
#     df = df[['inc', 'elem', '1_x', '2_x', '3_x', '1_eu', '2_eu', '3_eu','Phases', 'rho_m_point', 'rho_d_point', 'rho_point','Mises_strain','Mises_stress']]
#     df.columns = ['inc', 'elem', 'x', 'y', 'z', 'phi1', 'PHI', 'phi2','Phases', 'rho_m_point', 'rho_d_point', 'rho_point','Mises_strain','Mises_stress']
#     #
#     output = '%s_%s'%(geom_name,load_name)
#     file_rg = os.path.join(main_dir,'postProc','DAMASK_CA_%s_%s.txt'%(output,inci))
#     if not os.path.exists(os.path.join(main_dir,'postProc')):
#         os.makedirs(os.path.join(main_dir,'postProc'))
#     with open(file_rg,'w') as f:
#         df.to_string(f,index=False)
# if type == 'DAMASK_CA' and 1 == 2:
#     # intial geom data
#     rg = rgg.geom_regridder(geom_name,load_name)
#     grid0, size0, origin0, microstructure0_flatten, geom_comments0, skip = rg.read_geom(main_dir)
#     myH5 = p5.DADF5(hdf5_name)
#     df,df_nodal = myH5.make_displacements_df (inc=inci)
#     rebacked_id, new_grid = rg.report_restartGeom(df, df_nodal,CA=inci)
#     # file to open
#     hdf5_file_name = '%s_%s'%(geom_name,load_name)
#     file = os.path.join(main_dir, hdf5_file_name+'.hdf5')
#     f = h5py.File(file,'r')
#     #
#     inc_key = 'inc{:05}'.format(inci)
#     phase_name = myH5.constituents[0]
#     orientations = np.array(f[inc_key]['constituent'][phase_name]['generic']['orientation'])
#     orientations = np.array(orientations.tolist())
#     #
#     elem_rg  = int(new_grid.prod())
#     New_RVE_size = rg.New_RVE_size
#     #
#     ou = p5.outProc_util()
#     eulers = np.array(ou.make_eulersFromQuaternion(orientations))
#     new_eulers = []
#     for i, item in enumerate(rebacked_id):
#         new_eulers.append(eulers[item])
#     new_eulers = np.array(new_eulers)
#     #
#     Cell_coords=rg.create_grid(new_grid,New_RVE_size,origin0, point_type='Cell')
#     df = pd.DataFrame({'inc': inci*np.ones(elem_rg, dtype=int)})
#     df['elem'] = range(1,elem_rg+1)
#     df['1_x'] = Cell_coords[:,0]
#     df['2_x'] = Cell_coords[:,1]
#     df['3_x'] = Cell_coords[:,2]
#     #
#     df['1_eu'] = new_eulers[:,0]
#     df['2_eu'] = new_eulers[:,1]
#     df['3_eu'] = new_eulers[:,2]
#     #
#     rho_m = np.array(f[inc_key]['constituent'][phase_name]['plastic']['rho_mob'])
#     rho_d = np.array(f[inc_key]['constituent'][phase_name]['plastic']['rho_dip'])
#     rho = rho_m + rho_d
#     #
#     rho_m_new = []
#     rho_d_new = []
#     rho_new = []
#     for i, item in enumerate(rebacked_id):
#         rho_m_new.append(rho_m[item])
#         rho_d_new.append(rho_d[item])
#         rho_new.append(rho[item])
#     rho_m_new = np.array(rho_m_new)
#     rho_d_new = np.array(rho_d_new)
#     rho_new = np.array(rho_new)
#     #
#     #
#     for i in range(rho_m.shape[1]):
#         col_title = '%s_rho_m'%(i+1)
#         df[col_title] = rho_m_new[:,i]
#     for i in range(rho_m.shape[1]):
#         col_title = '%s_rho_d'%(i+1)
#         df[col_title] = rho_d_new[:,i]
#     for i in range(rho_m.shape[1]):
#         col_title = '%s_rho'%(i+1)
#         df[col_title] = rho_new[:,i]
#     mobile_list = [i for i in df.columns if i.split('_')[-2:]==['rho', 'm']]
#     df['rho_m_point'] = df[mobile_list].sum(axis=1)
#     dipole_list = [i for i in df.columns if i.split('_')[-2:]==['rho', 'd']]
#     df['rho_d_point'] = df[dipole_list].sum(axis=1)
#     dislo_list = [i for i in df.columns if i.split('_')[-1:]==['rho']]
#     df['rho_point'] = df[dislo_list].sum(axis=1)
#     #
#     #   ---- add a few parameter ----
#     #
#     #
#     #
#     #
#     df['Phases'] = np.ones(df.shape[0])
#     df = df[['inc', 'elem', '1_x', '2_x', '3_x', '1_eu', '2_eu', '3_eu','Phases', 'rho_m_point', 'rho_d_point', 'rho_point']]
#     df.columns = ['inc', 'elem', 'x', 'y', 'z', 'phi1', 'PHI', 'phi2','Phases', 'rho_m_point', 'rho_d_point', 'rho_point']
#     #
#     output = '%s_%s'%(geom_name,load_name)
#     file_rg = os.path.join(main_dir,'postProc','DAMASK_CA_%s_%s.txt'%(output,inci))
#     if not os.path.exists(os.path.join(main_dir,'postProc')):
#         os.makedirs(os.path.join(main_dir,'postProc'))
#     with open(file_rg,'w') as f:
#         df.to_string(f,index=False)
# elif type == 'postOut':
#     out_list = ['ori','f','fp','fe','p','e_Cauchy','s_Cauchy','rho','Mises']
#     out_list = ['eu','s_Cauchy','e_Cauchy','Mises','rho']
#     #
#     myH5 = p5.DADF5(hdf5_name)
#     ou = p5.outProc_util()
#     df,df_nodal = myH5.make_displacements_df (inc = inci)
#     df['1_u'] = df['1_avg(f).pos'] + df['1_fluct(f).pos']
#     df['2_u'] = df['2_avg(f).pos'] + df['2_fluct(f).pos']
#     df['3_u'] = df['3_avg(f).pos'] + df['3_fluct(f).pos']
#     #
#     f = h5py.File(hdf5_name,'r')
#     inc_key = 'inc{:05}'.format(inci)
#     phase_name = myH5.constituents[0]
#     #
#     # main outputs
#     orientations = np.array(f[inc_key]['constituent'][phase_name]['generic']['orientation'])
#     orientations = np.array(orientations.tolist())
#     #
#     F_def = np.array(f[inc_key]['constituent'][phase_name]['generic']['F'])
#     F_def = F_def.reshape((F_def.shape[0],9))
#     #
#     Pio = np.array(f[inc_key]['constituent'][phase_name]['generic']['P'])
#     Pio = Pio.reshape((Pio.shape[0],9))
#     if 'ori' in out_list:
#         for i in range(4):
#             col_title = '%s_orientation'%(i+1)
#             df[col_title] = orientations[:,i]
#     if 'f' in out_list:
#         for i in range(9):
#             col_title = '%s_f'%(i+1)
#             df[col_title] = F_def[:,i]
#     if 'fp' in out_list:
#         F_p = np.array(f[inc_key]['constituent'][phase_name]['generic']['Fp'])
#         F_p = F_p.reshape((F_p.shape[0],9))
#         for i in range(9):
#             col_title = '%s_fp'%(i+1)
#             df[col_title] = F_p[:,i]
#     if 'fe' in out_list:
#         F_e = np.array(f[inc_key]['constituent'][phase_name]['generic']['Fe'])
#         F_e = F_e.reshape((F_e.shape[0],9))
#         for i in range(9):
#             col_title = '%s_fe'%(i+1)
#             df[col_title] = F_e[:,i]
#     if 'p' in out_list:
#         for i in range(9):
#             col_title = '%s_p'%(i+1)
#             df[col_title] = Pio[:,i]
#     # output with calculations
#     if 'e_Cauchy' in out_list:
#         # add left cauchy strain
#         Cauchy_strain = np.array(ou.make_left_Cauchy_strain(F_def))
#         for i in range(9):
#             col_title = '%s_ln(V)'%(i+1)
#             df[col_title] = Cauchy_strain[:,i]
#     if 's_Cauchy' in out_list:
#         # add cauchy stress
#         cauchy_stress = np.array(ou.make_Cauchy_stress(F_def, Pio))
#         for i in range(9):
#             col_title = '%s_Cauchy'%(i+1)
#             df[col_title] = cauchy_stress[:,i]
#     if 'eu' in out_list:
#         # add eulers
#         eulers = np.array(ou.make_eulersFromQuaternion(orientations))
#         for i in range(3):
#             col_title = '%s_eu'%(i+1)
#             df[col_title] = eulers[:,i]
#     if 'Mises' in out_list:
#         # add eulers
#         mises = np.array(ou.Mises_calc_point(df))
#         df['Mises_strain'] = mises[:,1]
#         df['Mises_stress'] = mises[:,2]
#     if 'rho' in out_list:
#         # add dislocations
#         rho_m = np.array(f[inc_key]['constituent'][phase_name]['plastic']['rho_mob'])
#         rho_d = np.array(f[inc_key]['constituent'][phase_name]['plastic']['rho_dip'])
#         rho = rho_m + rho_d
#         for i in range(rho_m.shape[1]):
#             col_title = '%s_rho_m'%(i+1)
#             df[col_title] = rho_m[:,i]
#         for i in range(rho_m.shape[1]):
#             col_title = '%s_rho_d'%(i+1)
#             df[col_title] = rho_d[:,i]
#         for i in range(rho_m.shape[1]):
#             col_title = '%s_rho'%(i+1)
#             df[col_title] = rho[:,i]
#         mobile_list = [i for i in df.columns if i.split('_')[-2:]==['rho', 'm']]
#         df['rho_m_point'] = df[mobile_list].sum(axis=1)
#         dipole_list = [i for i in df.columns if i.split('_')[-2:]==['rho', 'd']]
#         df['rho_d_point'] = df[dipole_list].sum(axis=1)
#         dislo_list = [i for i in df.columns if i.split('_')[-1:]==['rho']]
#         df['rho_point'] = df[dislo_list].sum(axis=1)
#     output = '%s_%s'%(geom_name,load_name)
#     file_p = os.path.join(main_dir,'postProc','%s_%s.txt'%(output,inci))
#     if not os.path.exists(os.path.join(main_dir,'postProc')):
#         os.makedirs(os.path.join(main_dir,'postProc'))
#     with open(file_p,'w') as f:
#         f.write('1\theader\n')
#         # np.savetxt(f,data,delimiter='\t')
#         df.to_string(f,index=False)
#     file_p_nodal = os.path.join(main_dir,'postProc','%s_%s_nodal.txt'%(output,inci))
#     #
#     df_nodal['1_u'] = df_nodal['1_avg(f).pos'] + df_nodal['1_fluct(f).pos']
#     df_nodal['2_u'] = df_nodal['2_avg(f).pos'] + df_nodal['2_fluct(f).pos']
#     df_nodal['3_u'] = df_nodal['3_avg(f).pos'] + df_nodal['3_fluct(f).pos']
#     with open(file_p_nodal,'w') as f:
#         f.write('1\theader\n')
#         # np.savetxt(f,data,delimiter='\t')
#         df_nodal.to_string(f,index=False)
#
#
#
#
#
#
#
#
#
# #
#
# data = pd.read_csv('abc.txt', header = 0)
# cc=np.array(data['FeatureIds'])
# dd = list(set(cc))
# len(dd)
# a=np.array(data['FeatureIds'])
# geom = damask.Geom.from_file('63G.geom')
# b= a.reshape(geom.microstructure.shape,order = 'F')
#
# geom.microstructure = b
#
# geom.to_file('new.geom')
# #
