import numpy as np

def interior(arr, dim, grid):

  if (grid == "airfoil"):

    height  = int(dim[0])
    width   = int(dim[1])
    wwidth  = int(dim[4])
    dwidth  = int(dim[5])
 
    arr = arr.reshape( [height, width] )

    wall_rows_per_row = int(width/wwidth)
    wall_block_end = int(height/wall_rows_per_row)
    down_rows_per_row = int(width/dwidth)
    down_block_end = int(height/down_rows_per_row)

    end_block1=wall_block_end
    end_block2=2*wall_block_end
    end_block3 = end_block2+down_block_end
    end_block4= end_block3 + wall_block_end
    end_block5 = end_block4 + wall_block_end
    end_block6 = end_block5 + down_block_end

    block1 = arr[0:end_block1,:]
    b_1 = np.empty([0, wwidth], float)
    for i in range(0,block1.shape[0]):
     line = block1[i,:]
     line = line.reshape([wall_rows_per_row,wwidth])
     b_1  = np.append(b_1, line, axis=0)

    block2 = arr[end_block1:end_block2,:]
    b_2 = np.empty([0, wwidth], float)
    for i in range(0,block2.shape[0]):
     line = block2[i,:]
     line = line.reshape([wall_rows_per_row,wwidth])
     b_2  = np.append(b_2, line, axis=0)

    block3 = arr[end_block2:end_block3,:]
    b_3 = np.empty([0, dwidth], float)
    for i in range(0,block3.shape[0]):
     line = block3[i,:]
     line = line.reshape([down_rows_per_row,dwidth])
     b_3  = np.append(b_3, line, axis=0)

    block4 = arr[end_block3:end_block4,:]
    
    b_4 = np.empty([0, wwidth], float)
    for i in range(0,block4.shape[0]):
     line = block4[i,:]
     line = line.reshape([wall_rows_per_row,wwidth])
     b_4  = np.append(b_4, line, axis=0)

    block5 = arr[end_block4:end_block5,:]
    b_5 = np.empty([0, wwidth], float)
    for i in range(0,block5.shape[0]):
     line = block5[i,:]
     line = line.reshape([wall_rows_per_row,wwidth])
     b_5  = np.append(b_5, line, axis=0)

    block6 = arr[end_block5:end_block6,:]
    b_6 = np.empty([0, dwidth], float)
    for i in range(0,block6.shape[0]):
     line = block6[i,:]
     line = line.reshape([down_rows_per_row,dwidth])
     b_6  = np.append(b_6, line, axis=0)

    b_3 = np.flip(b_3, axis=1)
    b_4 = np.flip(b_4, axis=1)
    b_5 = np.flip(b_5, axis=1)
 
    ret = np.append(b_3,b_2, axis=1)
    ret = np.append(ret,b_1, axis=1)
    ret = np.append(ret,b_4, axis=1)
    ret = np.append(ret,b_5, axis=1)
    ret = np.append(ret,b_6, axis=1)

  elif (grid == "ellipse" or grid=="cylinder"):
     
    height = int(dim[0])
    width  = int(dim[1])

    w = int(width/4)

    arr = arr.reshape( [height, width] )

    b_1 = np.empty([0, w], float)
    b_2 = np.empty([0, w], float)
    b_3 = np.empty([0, w], float)
    b_4 = np.empty([0, w], float)

    for i in range(0,int(height/4)):
     line = arr[i,:]
     line = line.reshape([4,w])
     b_1  = np.append(b_1, line, axis=0)

    for i in range(int(height/4),2*int(height/4)):
     line = arr[i,:]
     line = line.reshape([4,w])
     b_2  = np.append(b_2, line, axis=0)
 
    for i in range(2*int(height/4),3*int(height/4)):
     line = arr[i,:]
     line = line.reshape([4,w])
     b_3  = np.append(b_3, line, axis=0)

    for i in range(3*int(height/4), 4*int(height/4)):
     line = arr[i,:]
     line = line.reshape([4,w])
     b_4  = np.append(b_4, line, axis=0)


    b_3 = np.flip(b_3, axis=1)
    b_4 = np.flip(b_4, axis=1)
 
    ret = np.append(b_3,b_4, axis=1)
    ret = np.append(ret,b_2, axis=1)
    ret = np.append(ret,b_1, axis=1)

  elif (grid == "1b_rect_grid"):

    height = int(dim[0])
    width  = int(dim[1])

    ret = arr.reshape( [height, width] )

  return ret


def boundary(arr, dim, grid, interior, bType):

  if (grid == "1b_o_grid"):
    
    b_1 = arr[0       :height]
    b_2 = arr[height  :2*height]
    b_3 = arr[2*height:3*height]
    b_4 = arr[3*height:4*height]
    
    b_1  = np.flip(b_1, axis=0)
    b_4  = np.flip(b_4, axis=0)
    
    ret = np.append(b_3, b_1)
    ret = np.append(ret, b_2)
    ret = np.append(ret, b_4)
    ret = ret.reshape([1,ret.size]) 


  elif (grid == "ellipse"):

    height = int(dim[0])

    b_1 = arr[0       :height]
    b_2 = arr[height  :2*height]
    b_3 = arr[2*height:3*height]
    b_4 = arr[3*height:4*height]
    
    b_3 = np.flip(b_3, axis=0)
    b_4  = np.flip(b_4, axis=0)
    
    ret = np.append(b_3, b_4)
    ret = np.append(ret, b_2)
    ret = np.append(ret, b_1)

    ret = ret.reshape([1,ret.size]) 

  elif (grid == "airfoil"):
 
    if (bType == "bottom"):
      width  = int(dim[1])
      wwidth = int(dim[4])
      dwidth = int(dim[5])

      b_1 = arr[0       :wwidth]
      b_2 = arr[wwidth  :2*wwidth]
      b_3 = arr[2*wwidth:3*wwidth]
      b_4 = arr[3*wwidth:4*wwidth]
      
      b_3 = np.flip(b_3, axis=0)
      b_4  = np.flip(b_4, axis=0)
      
      ret = np.append(b_2, b_1)
      ret = np.append(ret, b_3)
      ret = np.append(ret, b_4)
      
      int1 = interior[0,-dwidth:]
      int1 = np.flip(int1, axis=0)
      int2 = interior[0,:dwidth]
      int2 = np.flip(int2, axis=0)

      ret = np.append(int1,ret)
      ret = np.append(ret,int2)

      ret = ret.reshape([1,width]) 
      
    
    if (bType == "top"):
      width  = int(dim[1])
      wwidth = int(dim[4])
      dwidth = int(dim[5])
      b_1 = arr[0       :wwidth]
      b_2 = arr[wwidth  :2*wwidth]
      b_3 = arr[2*wwidth:2*wwidth+dwidth]
      b_4 = arr[2*wwidth+dwidth:3*wwidth+dwidth]
      b_5 = arr[3*wwidth+dwidth:4*wwidth+dwidth]
      b_6 = arr[4*wwidth+dwidth:4*wwidth+2*dwidth]
      
      b_3 = np.flip(b_3, axis=0)
      b_4  = np.flip(b_4, axis=0)
      b_5  = np.flip(b_5, axis=0)
      
      ret = np.append(b_3, b_2)
      ret = np.append(ret, b_1)
      ret = np.append(ret, b_4)
      ret = np.append(ret, b_5)
      ret = np.append(ret, b_6)
      

      ret = ret.reshape([1,width]) 

  return ret


