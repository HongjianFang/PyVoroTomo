#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : PyVoroTomo.py
#
# Purpose : Random projections based seismic tomography using PyKonal
#
# Creation Date : 22-10-2019
#
# Last Modified : Tue Oct 22 16:40:23 2019
#
# Created By : Hongjian Fang: hfang@mit.edu 
#
#_._._._._._._._._._._._._._._._._._._._._.*/

import yaml
from collections import defaultdict
import itertools
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from voronoicells import voronoicells

# read input file
if len(sys.argv) == 1:
    parafile = 'PyVoroTomo.in'
else:
    parafile = str(sys.argv[1])

with open(parafile,'r') as fin:
    par = yaml.load(fin,Loader=yaml.Fulloader)

#hvratio = par['hvratio']
latmin = par['latmin']
nlat = par['nlat']
lonmin = par['lonmin']
nlon = par['nlon']
depmin = par['depmin']
ndep = par['ndep']
dlat = par['dlat']
dlon = par['dlon']
ddep = par['ddep']
damp = par['damp']
datafile = par['datafile']
ncell = par['ncell']
nsets = par['nsets']

# read data
data = pd.HDFStore(datafile)
srcs = data['srcs']
arrs = data['arrs']

# selecting subsets of data
## come back here later for selecting based on spatial locations, i.e. downweight those inside event or station clusters
src_sub = src.sample(n=1000)
arrs_sub = arrs[arrs['event_id'].isin(srcsub['event_id'])]

# inversion
## constructing projection matrix
latmax = latmin+nlat*dlat
lonmax = lonmin+nlon*dlon
depmax = depmin+ndep*ddep
cellpos, nearcells = voronoicells(latmin=latmin,latmax=latmax,lonmin=lonmin,\
        lonmax=lonmax, depmin=depmin,depmax=depmax, ncell = ncell)

mdim = (nlat+1)*(nlon+1)*(ndep+1)
latgrid = np.linspace(latmin,latmax,nlat+1)
longrid = np.linspace(lonmin,lonmax,nlon+1)
depgrid = np.linspace(depmin,depmax,ndep+1)
colid = np.zeros(mdim,dtype=int)
rowid = np.zeros(mdim,dtype=int)
mmidx = 0
idx = 0
for kk in range(ndep+1):
    for jj in range(nlon+1):
        for ii in range(nlat+1):
            findpts = neiList[mmidx]-1
            dis = (latgrid[ii]-pos[findpts,0])**2+(longrid[jj]-pos[findpts,1])**2+\
                    (depgrid[kk]-pos[findpts,2])**2
            midx = np.argmin(dis)
            colid[idx] = findpts[midx]
            idxt = neiList[mmidx][midx]
            mmidx = int(idxt*np.sign(ii))
            rowid[idx] = idx
            idx += 1
Gp = csr_matrix((np.ones(mdim,),(rowid,colid)),shape=(mdim,ncell))

## decide whether to treat receivers as sources. Prefered if nrc << nsrc
rcs = arrs['station_id'].unique()
for irc in rcs:
    arr4rc = arrs_sub[arrs_sub['staion_id']==irc]
    pykonal.
    ttinterp
    for isrc in range(len(arr4rc)):
        src_pos = arrs_sub['ev'
        ray = solver.trace_ray(src_pos,step_size= ,tolerance=1e-1)
        dsyn = 
        ray = pd.DataFrame(ray,names=('lat','lon','depth'))
        ray['cellidx'] = ray.apply(lambda x:)
        nseg = ray['cellidx'].unique()

        ray = ray.groupby('cellidx').count()
        for iseg in range(len(ray)):
            colid = ray['cellidx'].iloc[iseg]
            rowid = ridx
            nonzero = step_size*ray['lat']

G = coo_matrix((nonzero,(rowid,colid)),shape=(ridx,ncell))
x = lsmr
vel = Gp*x
# output velocity model and locaitons

if __name__ == '__main__':
    main()
