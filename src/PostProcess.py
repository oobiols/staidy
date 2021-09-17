import os
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

class PostProcessAmr():
  def __init__(self, 
                n_bins=2,
	        patches=[],
		indices=[],
		true_data=np.ones((2,2)),
		patchheight=8,
		patchwidth=32,
		height=32,
		width=128,
	        case_name="channelflow",
                modelname = "amr",
                **kwargs):

    super(PostProcessAmr, self).__init__(**kwargs)

    self.n_bins = n_bins
    self.case_name = case_name
    self.patches = patches
    self.indices = indices
    self.height=height
    self.width = width
    self.patchheight = patchheight
    self.patchwidth = patchwidth
    self.npx = self.height // self.patchheight
    self.npy = self.width // self.patchwidth
    self.npatches = self.npx*self.npy
    self.modelname = modelname
    self.true_data = true_data

    self.total_n_cells = 0
    for patch in self.patches:
      n_patch = patch.shape[0]
      h = patch.shape[1]
      w = patch.shape[2]
      self.total_n_cells+= h*w*n_patch

  def field_to_png(self,variablename="xvelocity"):

    if (variablename == "xvelocity"):
     v = 0
    elif (variablename == "yvelocity"):
     v = 1
    elif (variablename == "pressure"):
     v = 2
    elif (variablename == "nutilda"):
     v = 3

    fig, axs = plt.subplots(self.npx, self.npy , gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(6,2))
    fig.suptitle(variablename+"_NNamr", fontsize=10)

    for i in range(self.npatches):
     for j , indices in enumerate(self.indices):

      if i in indices:
         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = self.patches[j]
         patch = patches[idx,:,:,:]

     x = i//self.npx
     y = i%self.npy

     data = patch[:,:,v]
     umin = np.min(self.true_data[0,:,:,v])
     umax = np.max(self.true_data[0,:,:,v])
     axs[x,y].set_xticks([])
     axs[x,y].set_yticks([])    
     axs[x,y].patch.set_edgecolor('black')  
     axs[x,y].patch.set_linewidth('1')
     hm = sn.heatmap(data, vmin = umin, vmax= umax, ax=axs[x,y],cbar=False, xticklabels=False,yticklabels=False)
 
    directory_name = './amr_fields/'+self.modelname
    file_name = variablename

    if not os.path.exists(directory_name):
      os.makedirs(directory_name)

    plt.savefig(directory_name+'/'+file_name,dpi=600)
    plt.close()
    
    plt.figure(figsize=(8,2))
    data= self.true_data[0,:,:,v]
    hm = sn.heatmap(data, vmin = umin, vmax= umax, xticklabels=False,yticklabels=False)
    plt.title(variablename+"_true")
    plt.savefig(directory_name+'/'+file_name+'_true',dpi=600)
    plt.close()


  def velocity_to_foam(self,uref=3.5):

    directory_name = './amr_to_foam/'+self.modelname
    file_name = "U"

    if not os.path.exists(directory_name):
      os.makedirs(directory_name)

    f = open(directory_name+'/'+file_name,'w')
    f.write('FoamFile\n{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tvolVectorField;\n\tobject\tU;\n\tlocation\t"1";\n}\n\n') 
    f.write('dimensions\t[0 1 -1 0 0 0 0];\n\n')
    f.write('internalField\tnonuniform List<vector>\n\n')
    f.write(str(self.total_n_cells)+'\n(\n')

    Ux = np.empty([0])
    Uy = np.empty([0])
    Uz = np.empty([0])

    for i in range(self.npatches):
     for j , indices in enumerate(self.indices):

      if i in indices:
         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = self.patches[j]
         patch = patches[idx,:,:,0:2]
         h = patch.shape[0]
         w = patch.shape[1]
         level = h//self.patchheight
         ux = patch[::level,::level,0].ravel()
         uz = patch[::level,::level,1].ravel()
         Ux = np.append(Ux,ux,axis=0)
         Uz = np.append(Uz,uz,axis=0)

    for i in range(self.npatches):
     for j , indices in enumerate(self.indices[1:]):

      if i in indices:

         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = self.patches[j+1]
         patch = patches[idx,:,:,0:2]

         h = patch.shape[0]
         w = patch.shape[1]
         level = h//self.patchheight
         ux = patch[::level,1::level,0].ravel()
         uz = patch[::level,1::level,1].ravel()
         Ux = np.append(Ux,ux,axis=0)
         Uz = np.append(Uz,uz,axis=0)
    
         ux = patch[1::level,::level,0].ravel()
         uz = patch[1::level,::level,1].ravel()
         Ux = np.append(Ux,ux,axis=0)
         Uz = np.append(Uz,uz,axis=0)

         ux = patch[1::level,1::level,0].ravel()
         uz = patch[1::level,1::level,1].ravel()
         Ux = np.append(Ux,ux,axis=0)
         Uz = np.append(Uz,uz,axis=0)

    for i in range(Ux.shape[0]):
        ux = Ux[i]*uref
        uy = 0
        uz = Uz[i]*uref

        f.write('('+str(ux)+' 0 '+str(uz)+')\n' )
    
    f.write(');\n\n')    

    f.write('boundaryField\n{\n')

    if self.case_name == "channelflow":

     f.write('\tinlet{\n\t\ttype\tfixedValue;\n\t\tvalue\tuniform ('+str(uref)+' 0 0);\n\t}\n\n') 
     f.write('\toutlet{\n\t\ttype\tzeroGradient;\n\t}\n\n') 
     f.write('\ttop{\n\t\ttype\tnoSlip;\n\t}\n\n') 
     f.write('\tbottom{\n\t\ttype\tnoSlip;\n\t}\n\n') 
     f.write('\tfrontAndBack{\n\t\ttype\tempty;\n\t}\n\n}') 

  def pressure_to_foam(self,uref=3.5):

    directory_name = './amr_to_foam/'+self.modelname
    file_name = "p"

    if not os.path.exists(directory_name):
      os.makedirs(directory_name)

    f = open(directory_name+'/'+file_name,'w')
    f.write('FoamFile\n{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tvolScalarField;\n\tobject\tp;\n\tlocation\t"1";\n}\n\n') 
    f.write('dimensions\t[0 2 -2 0 0 0 0];\n\n')
    f.write('internalField\tnonuniform List<scalar>\n\n')
    f.write(str(self.total_n_cells)+'\n(\n')

    P = np.empty([0])

    for i in range(self.npatches):
     for j , indices in enumerate(self.indices):

      if i in indices:
         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = self.patches[j]
         patch = patches[idx,:,:,2]
         h = patch.shape[0]
         w = patch.shape[1]
         level = h//self.patchheight
         p = patch[::level,::level].ravel()
         P = np.append(P,p,axis=0)

    for i in range(self.npatches):
     for j , indices in enumerate(self.indices[1:]):

      if i in indices:
         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = self.patches[j+1]
         patch = patches[idx,:,:,2]
         h = patch.shape[0]
         w = patch.shape[1]
         level = h//self.patchheight

         p = patch[::level,1::level].ravel()
         P = np.append(P,p,axis=0)
	
         p = patch[1::level,::level].ravel()
         P = np.append(P,p,axis=0)
    
         p = patch[1::level,1::level].ravel()
         P = np.append(P,p,axis=0)

    for i in range(P.shape[0]):
        p = P[i]*uref*uref

        f.write(str(p)+'\n' )
    
    f.write(');\n\n')    

    f.write('boundaryField\n{\n')

    if self.case_name == "channelflow":

     f.write('\tinlet{\n\t\ttype\tzeroGradient;\n\t}\n\n') 
     f.write('\toutlet{\n\t\ttype\tfixedValue;\n\t\tvalue\tuniform 0;\n\t}\n\n') 
     f.write('\ttop{\n\t\ttype\tzeroGradient;\n\t}\n\n') 
     f.write('\tbottom{\n\t\ttype\tzeroGradient;\n\t}\n\n') 
     f.write('\tfrontAndBack{\n\t\ttype\tempty;\n\t}\n\n}') 

  def nutilda_to_foam(self,nuref=3.5):

    directory_name = './amr_to_foam/'+self.modelname
    file_name = "nuTilda"

    if not os.path.exists(directory_name):
      os.makedirs(directory_name)

    f = open(directory_name+'/'+file_name,'w')
    f.write('FoamFile\n{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tvolScalarField;\n\tobject\tnuTilda;\n\tlocation\t"1";\n}\n\n') 
    f.write('dimensions\t[0 2 -1 0 0 0 0];\n\n')
    f.write('internalField\tnonuniform List<scalar>\n\n')
    f.write(str(self.total_n_cells)+'\n(\n')

    P = np.empty([0])

    for i in range(self.npatches):
     for j , indices in enumerate(self.indices):

      if i in indices:
         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = self.patches[j]
         patch = patches[idx,:,:,3]
         h = patch.shape[0]
         w = patch.shape[1]
         level = h//self.patchheight
         p = patch[::level,::level].ravel()
         P = np.append(P,p,axis=0)

    for i in range(self.npatches):
     for j , indices in enumerate(self.indices[1:]):

      if i in indices:
         idx = np.where(indices==i)
         idx = idx[0][0]
         patches = self.patches[j+1]
         patch = patches[idx,:,:,3]
         h = patch.shape[0]
         w = patch.shape[1]
         level = h//self.patchheight

         p = patch[::level,1::level].ravel()
         P = np.append(P,p,axis=0)

         p = patch[1::level,::level].ravel()
         P = np.append(P,p,axis=0)
    
         p = patch[1::level,1::level].ravel()
         P = np.append(P,p,axis=0)

    for i in range(P.shape[0]):
        p = P[i]*nuref

        f.write(str(p)+'\n' )
    
    f.write(');\n\n')    

    f.write('boundaryField\n{\n')

    if self.case_name == "channelflow":

     f.write('\tinlet{\n\t\ttype\tfixedValue;\n\tvalue\tuniform 3e-4;\n\t}\n\n') 
     f.write('\toutlet{\n\t\ttype\tzeroGradient;\n\t}\n\n') 
     f.write('\ttop{\n\t\ttype\tfixedValue;\n\tvalue\tuniform 0;\n\t}\n\n') 
     f.write('\tbottom{\n\t\ttype\tfixedValue;\n\tvalue\tuniform 0;\n\t}\n\n') 
     f.write('\tfrontAndBack{\n\t\ttype\tempty;\n\t}\n\n}') 

