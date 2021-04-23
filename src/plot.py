import matplotlib.pyplot as plt
import numpy as np
import settings
import os

from metrics import *

def history(history,name, writing=1):

 title = "history-"+name
 path="./histories/"
 if not os.path.exists(path):
    os.makedirs(path)
 plt.figure()
 plt.title(title)
 

 for n, values in history.history.items():

  n = str(n)
  if n=="loss" or n=="val_loss" or n=="data_loss" or n=="pde_loss" or n=="val_data_loss" or n=="val_pde_loss":
   plt.plot(values,label=n)

  if(writing):

   path="./losses/"
   if not os.path.exists(path):
    os.makedirs(path)

   filename = n+"-"+name+".txt"
   f = open(path+filename,"w")
   for value in values:
    f.write(str(value)+"\n")
 

 plt.legend() 
 plt.ylim(1e-7,1)
 plt.xlabel("epoch")
 plt.ylabel("loss")
 plt.yscale("log")
 plt.savefig('./histories/'+title+'.jpg')
 plt.close()

 return 0


