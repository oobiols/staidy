import numpy as np


def timeStep(Ux, Uy, Uavg, p, pplus, nut, nutAvg, UxTop,UyTop,pTop,nutTop,case, turb):

 if(case=="ellipse"):

## VELOCITY 
   s = Ux.shape[0]
   s = str(s)
   f = open("./printedFields/U", "w")
   f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volVectorField;\n	location	0;\n	object	U;\n}\n")
   f.write("dimensions [0 1 -1 0 0 0 0];\n\n")
   f.write("internalField\t nonuniform List<vector>\n" + s + "\n(\n")
   for j in range(0,int(s)): 
    f.write("("+repr(Ux[j]*Uavg)+" 0 "+repr(Uy[j]*Uavg)+")\n")
   f.write(");\n")
   f.write("boundaryField\n{\n")
   f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform (0.6 0 0);\n}")
   f.write("\nbottom\n{\n\ttype\t noSlip;\n}")
   f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")
   f.write("}\n")

# PRESSURE  
   s = Ux.shape[0]
   s = str(s)
   f = open("./printedFields/p", "w")
   f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	object	p;\n}\n")
   f.write("dimensions [0 2 -2 0 0 0 0];\n\n")
   f.write("internalField\t nonuniform List<scalar>\n" + s + "\n(\n")
   for j in range(0,int(s)): 
    f.write(repr(p[j]*(Uavg*Uavg)/pplus) +"\n")
   
   f.write(");\n")
   f.write("boundaryField\n{\n")
   f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform 0;\n}")
   f.write("\nbottom\n{\n\ttype\t zeroGradient;\n}")
   f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

   f.write("}\n")
 
   if (turb == 1):

     s = Ux.shape[0]
     s = str(s)
     f = open("./printedFields/nuTilda", "w")
     f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	object	nuTilda;\n}\n")
     f.write("dimensions [0 2 -1 0 0 0 0];\n\n")
     f.write("internalField\t nonuniform List<scalar>\n" + s + "\n(\n")
     for j in range(0,int(s)): 
      f.write(repr(nut[j]*(nutAvg))+"\n")
     
     f.write(");\n")
     s=str(s)
     f.write("boundaryField\n{\n")
     f.write("\ntop\n{\n\ttype\t freestream;\n\tfreestreamValue\t uniform 3e-6;\n}")
  
     f.write("\nbottom\n{\n\ttype\t fixedValue;\nvalue\t uniform 0;}")
     f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")
  
     f.write("}\n")


 if (case == "channelFlow"):

## VELOCITY 
 
   s = Ux.shape[0]
   s = str(s)
   f = open("./printedFields/U", "w")
   f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volVectorField;\n	location	0;\n	object	U;\n}\n")
   f.write("dimensions [0 1 -1 0 0 0 0];\n\n")
   f.write("internalField\t nonuniform List<vector>\n" + s + "\n(\n")
   for j in range(0,int(s)): 
    f.write("("+repr(Ux[j]*Uavg)+" "+repr(Uy[j]*Uavg)+" 0)\n")
   
   f.write(");\n")
 
   f.write("boundaryField\n{\n")
   f.write("inlet\n{\n\ttype\t fixedValue;\n\tvalue\t uniform("+repr(Uavg)+ " 0 0);\n}")
   f.write("\noutlet\n{\n\ttype\t zeroGradient;\n}")
   f.write("\ntop\n{\n\ttype\t noSlip;\n}")
   f.write("\nbottom\n{\n\ttype\t noSlip;\n}")
   f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

   f.write("}\n")

# PRESSURE  
 
   s = Ux.shape[0]
   s = str(s)
   f = open("./printedFields/p", "w")
   f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	object	p;\n}\n")
   f.write("dimensions [0 2 -2 0 0 0 0];\n\n")
   f.write("internalField\t nonuniform List<scalar>\n" + s + "\n(\n")
   for j in range(0,int(s)):
 
    f.write(repr(p[j]*(Uavg*Uavg)/pplus) +"\n")
   
   f.write(");\n")
 
   f.write("boundaryField\n{\n")
   f.write("outlet\n{\n\ttype\t fixedValue;\n\tvalue\t uniform 0;\n}")
   f.write("\ninlet\n{\n\ttype\t zeroGradient;\n}")
   f.write("\ntop\n{\n\ttype\t zeroGradient;\n}")
   f.write("\nbottom\n{\n\ttype\t zeroGradient;\n}")
   f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")

   f.write("}\n")

   if (turb == 1):

     s = Ux.shape[0]
     s = str(s)
     f = open("./printedFields/nuTilda", "w")
     f.write("FoamFile\n{\n	version 	2.0;\n	format	ascii;\n  class volScalarField;\n	location	0;\n	object	nuTilda;\n}\n")
     f.write("dimensions [0 2 -1 0 0 0 0];\n\n")
     f.write("internalField\t nonuniform List<scalar>\n" + s + "\n(\n")
     for j in range(0,int(s)): 
      f.write(repr(nut[j]*(nutAvg))+"\n")
     
     f.write(");\n")
   
     f.write("boundaryField\n{\n")
     f.write("inlet\n{\n\ttype\t fixedValue;\n\tvalue\t uniform 0.001;\n}")
     f.write("\noutlet\n{\n\ttype\t zeroGradient;\n}")
     f.write("\ntop\n{\n\ttype\t fixedValue;\n\tvalue\t uniform 0;\n}")
     f.write("\nbottom\n{\n\ttype\t fixedValue;\n\tvalue\t uniform 0;\n}")
     f.write("\nfrontAndBack\n{\n\ttype\t empty;\n}\n")
  
     f.write("}\n")
