def caldtdx(ray):
  # pseudo code for calculating derivative of dt with respect to hypocenter
  # ray is the ray path, 3D array (rad,lon,lat) with lat and lon in rad
  dpos = np.zeros(4,)
  # ray from station to event
  dpos[0] = ray[-1,0]-ray[-2,0]
  dpos[1] = ray[-1,0]*(ray[-1,1]-ray[-2,1])
  dpos[2] = ray[-1,0]*(ray[-1,2]-ray[-2,2])*np.cos(ray[-1,1])
  
  # this is a unit vector point to the direction of the ray path (first two points in a ray)
  dpos = dpos/np.sqrt(np.sum(dpos**2))
  # interpolate velocity model to the hypocenter point (rad0,lon0,lat0)
  vel_hypo = interpolate(vel,rad0,lon0,lat0)
  dtdx = np.zeros(4,)
  # the derivative is just the projection of slowness to different direction
  dtdx[:-1] = dpos/vel_hypo
  dtdx[-1] = -1.0
  return dtdx
