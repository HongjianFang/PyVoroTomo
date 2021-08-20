#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : generateinit.py
#
# Purpose :
#
# Creation Date : 13-05-2020
#
# Last Modified : Tue 10 Aug 2021 05:00:24 PM CST
#
# Created By : Hongjian Fang: hfang@mit.edu
#
#_._._._._._._._._._._._._._._._._._._._._.*/
import pandas as pd
import pykonal
import numpy as np
from scipy.interpolate import interp1d
radial = 6371.0

# change according to your case
lats = 32.2-0.1
latn = 36.8+0.1
lonw = -120.3-0.1
lone = -114.7+0.1
dep0 = -4.0
dep1 = 40.0
ndepth = 40
nlat = 200
nlon = 250
vel1dfile = 'cvms1d.csv'

#for P model
pinit = pykonal.fields.ScalarField3D(coord_sys='spherical')
npts = np.array([ndepth,nlat,nlon])
pinit.npts = npts
pinit.min_coords = [radial-dep1,np.pi/2-np.deg2rad(latn),np.deg2rad(lonw)]
pinit.node_intervals = [(dep1-dep0),np.deg2rad(latn-lats),np.deg2rad(lone-lonw)]/(npts-1)
pvalues = np.zeros(npts)

scec1d = pd.read_csv(vel1dfile)
scec1d['dep'].iloc[0] = -5.0
print(scec1d.head())
f = interp1d(radial-scec1d['dep'],scec1d['vp'])
vpinterp = f(radial-np.linspace(dep1,dep0,npts[0]))

for ii in range(npts[0]):
    pvalues[ii,:,:] = vpinterp[ii]

pinit.values = pvalues

pinit.to_hdf('initial_pwave_model.h5')


#for S model
pinit = pykonal.fields.ScalarField3D(coord_sys='spherical')
pinit.npts = npts
pinit.min_coords = [radial-dep1,np.pi/2-np.deg2rad(latn),np.deg2rad(lonw)]
pinit.node_intervals = [(dep1-dep0),np.deg2rad(latn-lats),np.deg2rad(lone-lonw)]/(npts-1)
pvalues = np.zeros(npts)

f = interp1d(radial-scec1d['dep'],scec1d['vs'])
vpinterp = f(radial-np.linspace(dep1,dep0,npts[0]))

for ii in range(npts[0]):
    pvalues[ii,:,:] = vpinterp[ii]

pinit.values = pvalues

pinit.to_hdf('initial_swave_model.h5')
