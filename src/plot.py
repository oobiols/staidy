import matplotlib.pyplot as plt
import numpy as np
import settings
import os


def history(history,name, writing=1):

 title = "history-"+name
 path="./histories/"
 if not os.path.exists(path):
    os.makedirs(path)
 plt.figure(dpi=800)
 plt.title(title)
 

 losses = ["loss","val_loss","data_loss","val_data_loss","cont_loss","val_cont_loss","mom_x_loss","val_mom_x_loss","mom_z_loss","val_mom_z_loss"]
 for n, values in history.history.items():

  n = str(n)

  if n in losses:
   plt.plot(values,label=n)

  if(writing):

   path="./losses/"
   if not os.path.exists(path):
    os.makedirs(path)

   filename = n+"-"+name+".txt"
   f = open(path+filename,"w")
   for value in values:
    f.write(str(value)+"\n")
 

 plt.legend(prop={'size': 6}) 
 plt.ylim(1e-8,10)
 plt.xlabel("epoch")
 plt.ylabel("loss")
 plt.yscale("log")
 plt.grid(b=True,which='both',axis='both')
 plt.savefig('./histories/'+title+'.png')
 plt.close()

 return 0


