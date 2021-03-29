import mapping
import openfoamparser as Ofpp
import matplotlib.pyplot as plt
import numpy as np
import math

  
def interiorData(addr, turb):

   U = np.float32(Ofpp.parse_internal_field(addr+"/U"))
   p = np.float32(Ofpp.parse_internal_field(addr+"/p"))

   if (turb):
    nuTilda     = np.float32(Ofpp.parse_internal_field(addr+"/nuTilda"))
   else:
    nuTilda = 0

   Ux          = U[:,0]
   Uy          = U[:,2]

   return Ux, Uy, p, nuTilda

def boundaryData(addr, turb): 

   zone        = vtki.PolyData(addr)
   U           = zone.cell_arrays['U']
   p           = zone.cell_arrays['p'] 

   if (turb):
    nuTilda     = zone.cell_arrays['nuTilda']
   else:
    nuTilda = 0
   
   Ux          = U[:,0]
   Uy          = U[:,2]

   return Ux, Uy, p, nuTilda


def single_sample(grid, interior_addr, bottom_addr, top_addr, dim, turb, pos):

  height = int(dim[0])
  length = dim[2]
  visc = dim[3]

  Ux_interior, Uy_interior, p_interior, nuTilda_interior = interiorData(interior_addr, turb)

#  Ux_bottom,   Uy_bottom,   p_bottom, nuTilda_bottom     = boundaryData(bottom_addr, turb)
#  Ux_top,      Uy_top,      p_top, nuTilda_top           = boundaryData(top_addr, turb)

 # Ux_bottom,   Uy_bottom,   p_bottom, nuTilda_bottom     = boundaryData(interior_addr, turb)
 # Ux_top,      Uy_top,      p_top, nuTilda_top           = boundaryData(interior_addr, turb)

  Ux_interior = mapping.interior(Ux_interior, dim, grid)
  Uy_interior = mapping.interior(Uy_interior, dim, grid)
  p_interior  = mapping.interior(p_interior, dim, grid)

  Ux = Ux_interior
  Uy = Uy_interior
  p  = p_interior

#  if pos == "input":

#   boundary="bottom"
#   Ux_bottom = mapping.boundary(Ux_bottom, dim, grid, Ux_interior, boundary)
#   Uy_bottom = mapping.boundary(Uy_bottom, dim, grid, Uy_interior, boundary)
#   p_bottom  = mapping.boundary(p_bottom,  dim, grid, p_interior, boundary)

#   boundary="top"
#   Ux_top = mapping.boundary(Ux_top, dim, grid, Ux_interior, boundary)
#   Uy_top = mapping.boundary(Uy_top, dim, grid, Uy_interior, boundary)
#   p_top  = mapping.boundary(p_top,  dim, grid, p_interior, boundary)

  if (turb):

   nuTilda_interior = mapping.interior(nuTilda_interior, dim, grid)
   nuTilda = nuTilda_interior

#   if pos == "input":
#    boundary="bottom"
# #   nuTilda_bottom = mapping.boundary(nuTilda_bottom, dim, grid,nuTilda_interior,boundary)
#    boundary="top"
#  #  nuTilda_top = mapping.boundary(nuTilda_top, dim, grid, nuTilda_interior,boundary)
#
#  if pos == "input":
#
# #  Ux = np.append(Ux_bottom, Ux_interior, axis = 0)
# #  Uy = np.append(Uy_bottom, Uy_interior, axis = 0)
# #  p  = np.append(p_bottom, p_interior,   axis = 0) 
#
#   if (turb): 
#  #  nuTilda = np.append(nuTilda_bottom, nuTilda_interior, axis = 0)
#  
#  # Ux = np.append(Ux, Ux_top,  axis = 0)
# #  Uy = np.append(Uy, Uy_top,  axis = 0)
#  # p  = np.append(p,  p_top,   axis = 0) 
#
#   if (turb):
#   # nuTilda = np.append(nuTilda, nuTilda_top,  axis = 0)

  Ux = Ux.reshape([Ux.shape[0], Ux.shape[1], 1])
  Uy = Uy.reshape([Uy.shape[0], Uy.shape[1], 1]) 
  p = p.reshape( [p.shape[0],  p.shape[1],  1])
  if (turb):
   nuTilda = nuTilda.reshape([nuTilda.shape[0], nuTilda.shape[1], 1])

  if (grid == "1b_rect_grid"):
    Ux_avg = Ux[int(height/2),0,0]
    Uy_avg = Uy[int(height/2),0,0]
    Uavg = Ux_avg
    nuTildaAvg = dim[3]

  elif (grid == "ellipse"):
    #mainux = Ux[height+1,0,0]
    #mainuy = Uy[height+1,0,0]
    uavg = 0.6
    nuTildaAvg = dim[3]

  elif (grid == "airfoil"):
    Uavg = Ux[height+1,0,0]
    nuTildaAvg = dim[3]
  
  if pos == "input" or pos == "output":

    Ux /= uavg
    Uy /= uavg
    p /= uavg*uavg

    if (turb): 
     nuTilda /= nuTildaAvg


  data    = np.concatenate( (Ux, Uy) , axis=2)  
  data    = np.concatenate( (data, p), axis=2)

  if (turb):
   data   = np.concatenate( ( data, nuTilda), axis=2) 

  return data


def case_data       (x_addrs, y_addr, coordinates, dim, grid, turb, x_train, y_train):

  
  x_addrs = x_addrs[0]
  n = len(x_addrs)

  y_interior_addr = y_addr[0]
  y_interior_addr = y_interior_addr[0]
  y_top_addr = y_interior_addr
  y_bottom_addr = y_interior_addr

  for i in range(0,n):

    x_interior_addr = x_addrs[i]
    x_bottom_addr   = []
    x_top_addr      = []

    pos = "input"
    data_cell  = single_sample(grid,     x_interior_addr, 
                              x_bottom_addr, x_top_addr,
			      dim, turb, pos)
    
    x_train.append(data_cell)

    pos = "output"
    data_cell = single_sample(grid,     y_interior_addr,  
		              y_bottom_addr, y_top_addr,
			      dim, turb, pos)


    y_train.append(data_cell)

  
  return

