import sys
sys.path.insert(0, './func/')

import numpy as np
import post
import writeToFoam

turb=1
i=25
i = str(i)
Uavg=(0.6)
pplus=1
visc=1e-4
case="ellipse"
nut=0
height=512
width = 512
wwidth=32
dwidth=64

Ux = np.loadtxt("./results/fields/Ux_cell_int_"+i+".out", delimiter=',')
Uy = np.loadtxt("./results/fields/Uy_cell_int_"+i+".out", delimiter=',')
p  = np.loadtxt("./results/fields/p_cell_int_"+i+".out",  delimiter=',')
if (turb):
 nut  = np.loadtxt("./results/fields/nut_cell_int_"+i+".out",  delimiter=',')


dim = np.array([height,width, wwidth,dwidth])
Ux_int = post.interiortoFoam(Ux,case,dim)
Uy_int = post.interiortoFoam(Uy, case, dim)
p_int  = post.interiortoFoam(p, case, dim)
if (turb):
 nut_int  = post.interiortoFoam(nut,case, dim)


#Ux = np.loadtxt("./results/fields/Ux_cell_top_"+i+".out", delimiter=',')
#Uy = np.loadtxt("./results/fields/Uy_cell_top_"+i+".out", delimiter=',')
#p  = np.loadtxt("./results/fields/p_cell_top_"+i+".out",  delimiter=',')
#if(turb):
# nut  = np.loadtxt("./results/fields/nut_cell_top_"+i+".out",  delimiter=',')
#
#Ux_top = post.boundarytoFoam(Ux, case)
#Uy_top = post.boundarytoFoam(Uy,case)
#p_top  = post.boundarytoFoam(p,case)
#if (turb):
# nut_top  = post.boundarytoFoam(nut,case)

Ux_top=0
Uy_top=0
nut_top=0
p_top=0
writeToFoam.timeStep(Ux_int,Uy_int,Uavg,p_int,pplus,nut_int,visc,Ux_top,Uy_top,p_top,nut_top,case,turb)




