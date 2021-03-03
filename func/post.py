import numpy as np

def boundarytoFoam(arr, case):

 if case == "channelFlow": 
  
    arr = arr

 elif case == "airfoil":

    height = int(arr.shape[0]/4)

    b_12 = arr[0:2*height]
    b_34 = arr[2*height:4*height]
  
    b_12 = np.flip(b_12, axis=0)
    b_34 = np.flip(b_34, axis=0)   
 
    arr = np.append(b_12,b_34)

 elif case == "ovals":

    height = int(arr.shape[0]/4)

    b_3 = arr[0:height]
    b_1 = arr[height:2*height]
    b_2 = arr[2*height:3*height]
    b_4 = arr[3*height:4*height]

    b_1 = np.flip(b_1,axis=0)
    b_4 = np.flip(b_4,axis=0)

    arr = np.append(b_1,b_2)
    arr = np.append(arr,b_3)
    arr = np.append(arr,b_4)

 elif case == "ellipse":

    w = int(arr.shape[0]/4)
    
    b_3 = arr[0       :w]
    b_4 = arr[w  :2*w]
    b_2 = arr[2*w:3*w]
    b_1 = arr[3*w:4*w]
    
    b_3 = np.flip(b_3, axis=0)
    b_4  = np.flip(b_4, axis=0)
    
    arr = np.append(b_1, b_2)
    arr = np.append(arr, b_3)
    arr = np.append(arr, b_4)

 return arr

def interiortoFoam(arr, case, dim):

 if (case == "airfoil"):

    height = int(dim[0])
    width  = int(dim[1])
    wwidth = int(dim[2])
    dwidth = int(dim[3])
    
    b1 = arr[:,dwidth+wwidth:dwidth+2*wwidth]
    b_1 = np.empty([0, width], float)
    jump = int(width/wwidth)
    for i in range (0,height,jump):

     line1 = b1[i:i+jump,:]
     line1 = line1.reshape([1,width])
     b_1 = np.append(b_1,line1,axis=0)

    b2 = arr[:,dwidth:dwidth+wwidth]
    b_2 = np.empty([0, width], float)
    jump = int(width/wwidth)
    for i in range (0,height,jump):

     line2 = b2[i:i+jump,:]
     line2 = line2.reshape([1,width])
     b_2 = np.append(b_2,line2,axis=0)

    b3 = arr[:,:dwidth]
    b3 = np.flip(b3,axis=1)
    b_3 = np.empty([0, width], float)
    jump = int(width/dwidth)
    for i in range (0,height,jump):

     line3 = b3[i:i+jump,:]
     line3 = line3.reshape([1,width])
     b_3 = np.append(b_3,line3,axis=0)

    b4 = arr[:,dwidth+2*wwidth:dwidth+3*wwidth]
    b4 = np.flip(b4,axis=1)
    b_4 = np.empty([0, width], float)
    jump = int(width/wwidth)
    for i in range (0,height,jump):

     line4 = b4[i:i+jump,:]
     line4 = line4.reshape([1,width])
     b_4 = np.append(b_4,line4,axis=0)
 
    b5 = arr[:,dwidth+3*wwidth:dwidth+4*wwidth]
    b5 = np.flip(b5,axis=1)
    b_5 = np.empty([0, width], float)
    jump = int(width/wwidth)
    for i in range (0,height,jump):

     line5 = b5[i:i+jump,:]
     line5 = line5.reshape([1,width])
     b_5 = np.append(b_5,line5,axis=0)

    b6 = arr[:,dwidth+4*wwidth:2*dwidth+4*wwidth]
    b_6 = np.empty([0, width], float)
    jump = int(width/dwidth)
    for i in range (0,height,jump):

     line6 = b6[i:i+jump,:]
     line6 = line6.reshape([1,width])
     b_6 = np.append(b_6,line6,axis=0)

    
    arr = np.append(b_1,b_2, axis=0)
    arr = np.append(arr,b_3, axis=0)
    arr = np.append(arr,b_4, axis=0)
    arr = np.append(arr,b_5, axis=0)
    arr = np.append(arr,b_6, axis=0)
    arr = arr.reshape([height * width])

 elif case == "channelFlow": 
    h = arr.shape[0]
    w = arr.shape[1]
    arr = arr.reshape( [h*w,1] )

    arr = np.array(arr).flatten()

 elif case == "ellipse":

    height = int(arr.shape[0])
    width = int(arr.shape[1])
    w = int(width/4)

    b_1 = np.empty([0, width], float)
    b_2 = np.empty([0, width], float)
    b_3 = np.empty([0, width], float)
    b_4 = np.empty([0, width], float)


    for i in range (0,height,4):

     line1 = arr[i:i+4,3*w:4*w]
     line1 = line1.reshape([1,width])
     b_1 = np.append(b_1,line1,axis=0)
 
     line2 = arr[i:i+4,2*w:3*w]
     line2 = line2.reshape([1,width])
     b_2 = np.append(b_2,line2,axis=0)
     
     line4 = arr[i:i+4,w:2*w]
     line4 = np.flip(line4,axis=1)
     line4 = line4.reshape([1,width])
     b_4 = np.append(b_4,line4,axis=0)

     line3 = arr[i:i+4,0:w]
     line3 = np.flip(line3,axis=1)
     line3 = line3.reshape([1,width])
     b_3 = np.append(b_3,line3,axis=0)
    
    b_1 = b_1.reshape([height * w])
    b_2 = b_2.reshape([height * w])
    b_3 = b_3.reshape([height * w])
    b_4 = b_4.reshape([height * w])

    
    arr = np.append(b_1,b_2)
    arr = np.append(arr,b_3)
    arr = np.append(arr,b_4)

    

 return arr


