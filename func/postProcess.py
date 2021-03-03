import numpy as np

import matplotlib.pyplot as plt
import post

def save_nondim_predicted_fields(y_predict, height, CNN, turb):

 n_pred = y_predict.shape[0]
 
 for i in range(0, n_pred):

  Ux_pred_int = y_predict[i,:,:,0]

  Uy_pred_int = y_predict[i,:,:,1]

  p_pred_int  = y_predict[i,:,:,2]
 
  if(turb):
   nut_pred_int  = y_predict[i,:,:,3]


  np.savetxt('./results/fields/Ux_cell_int_'+str(i)+'.out',Ux_pred_int, delimiter=',') 
  np.savetxt('./results/fields/Uy_cell_int_'+str(i)+'.out',Uy_pred_int, delimiter=',')
  np.savetxt('./results/fields/p_cell_int_'+str(i)+'.out', p_pred_int, delimiter=',')
  if(turb):
   np.savetxt('./results/fields/nut_cell_int_'+str(i)+'.out', nut_pred_int, delimiter=',')

