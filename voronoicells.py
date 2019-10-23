#/* -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#
# File Name : voronoicells.py
#
# Purpose :
#
# Creation Date : 22-10-2019
#
# Last Modified : Tue Oct 22 16:40:08 2019
#
# Created By : Hongjian Fang: hfang@mit.edu 
#
#_._._._._._._._._._._._._._._._._._._._._.*/

def voronoicells(latmin=0,latmax=10,lonmin=0,lonmax=10,\
        depmin=0,depmax=30, ncell = 300):
    pos = np.zeros((ncell,3))
    #print(iset)
    phi = np.random.rand(ncell,)*(latmax-latmin)+latmin
    phi = np.pi/2-np.deg2rad(phi)
    theta = np.random.rand(ncell,)*(lonmax-lonmin)+lonmin
    theta = np.deg2rad(theta)
    rad = 6371.0-depmax-np.random.rand(ncell,)*(depmax-depmin)+depmin

    xpts = rad*np.sin(phi)*np.cos(theta)
    ypts = rad*np.sin(phi)*np.sin(theta)
    zpts = rad*np.cos(phi)
    pos[:,0] = xpts
    pos[:,1] = ypts 
    pos[:,2] = zpts 
    tri = Delaunay(pos)
    neiList=defaultdict(list)
    for p in tri.vertices:
        for i,j in itertools.combinations(p,2):
            neiList[i+1].append(j+1)
            neiList[j+1].append(i+1)
    for p in range(1,1+len(pos)):
        neiList[0].append(p)
    neiList[0] = np.unique(neiList[0])
    for p in range(1,len(pos)+1):
        neiList[p].append(p)
        neiList[p] = np.unique(neiList[p])
    return pos,neiList
