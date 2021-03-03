import matplotlib.pyplot as plt
import numpy as np
import settings
import os

from metrics import *

def history(history,name, writing=1):

 title = "history-"+name
 plt.figure()
 plt.title(title)
 

 for n, values in history.history.items():

  n = str(n)
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
 plt.ylim(1e-5,1)
 plt.xlabel("epoch")
 plt.ylabel("loss")
 plt.yscale("log")
 plt.savefig('./tempHistories/'+title+'.jpg')
 plt.close()

 return 0


