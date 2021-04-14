import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

class NSDataSet:
  def __init__(self):
    self.fNames  = []
    self.nTotalGenome = 0
    self.var = np.zeros((1,1,1,3)) # u, v, p
    self.Re  = np.zeros((1,))
    self.xy  = np.zeros((1,1,1,2))
    self.loc = np.zeros((1,2))
    self.gSize = np.zeros(2)
    self.isLoaded = False
    self.shuffledIndices = None
    self.bounds = np.zeros((3,2))
    self.nColPnt = 400
    self.xyCol   = np.zeros((1,1,2))
    self.bc      = np.zeros((1,1))
    self.xyBc    = np.zeros((1,1))

  def add_file(self, fName):
    self.fNames.append(fName)
    self.isLoaded = False

  def load_data(self, preprocess=True, shuffle=True):
    assert len(self.fNames) > 0
    dataFile   = h5.File(self.fNames[0], 'r')
    self.gSize = dataFile.attrs['genomeSize']
    # get total number of genomes in all files
    for fName in self.fNames:
      dataFile = h5.File(fName, 'r')
      self.nTotalGenome += dataFile.attrs['nGenome']
      assert self.gSize[0] == dataFile.attrs['genomeSize'][0]
      assert self.gSize[1] == dataFile.attrs['genomeSize'][1]
    # allocate data for variables and coordinates
    self.xy  = np.zeros((self.nTotalGenome, self.gSize[0], self.gSize[1], 2))
    self.var = np.zeros((self.nTotalGenome, self.gSize[0], self.gSize[1], 3))
    # self.loc = np.zeros((self.nTotalGenome, 2))
    # self.Re  = np.zeros(self.nTotalGenome)
    # load cooridates and variables
    s, e = 0, 0
    for fName in self.fNames:
      dataFile = h5.File(fName, 'r')
      e = s + dataFile.attrs['nGenome']
      self.xy[s:e, :, :, :]  = dataFile['Coordinates'][...]
      # self.Re[s:e]           = dataFile['Re'][...]
      self.var[s:e, :, :, :] = dataFile['Variables'][...]
      # self.loc[s:e,:]        = dataFile['location'][...]
      s = e
    # preprocess
    if preprocess: self.preprocess()
    # shuffle
    if shuffle:
      self.shuffledIndices = np.arange(self.nTotalGenome)
      np.random.shuffle(self.shuffledIndices)
      self.xy[...]  = self.xy[self.shuffledIndices,...]
      self.var[...] = self.var[self.shuffledIndices,...]
      # self.loc[...] = self.loc[self.shuffledIndices,...]
      # self.Re[...]  = self.Re[self.shuffledIndices]
    # allocate bc, xyBc used for analysis
    nBcPoint  = 2*(self.gSize[0] + self.gSize[1] - 2)
    self.bc   = np.zeros((self.nTotalGenome, nBcPoint,3))
    self.xyBc = np.zeros((self.nTotalGenome, nBcPoint,2))
    #
    self.isLoaded = True

  def preprocess(self):
    '''
    translate cooridates by making the xmin and ymin both zero
    normalize u, v, and p, assume nu = 1.0e-3 for all cases
    '''
    # for g in range(self.nTotalGenome):
      # # translate coordinates
      # # x, y are generated in such a way that the min values >= 0
      # self.xy[g, :, :, 0] -= np.amin(self.xy[g, :, :, 0])
      # self.xy[g, :, :, 1] -= np.amin(self.xy[g, :, :, 1])
      # ## substract by the min pressure
      # #self.var[g, :, :, 2] -= np.amin(self.var[g, :, :, 2])
      # # normalize variables, using character velocity and length (l=1)
      # uRef = self.Re[g] * 1.0e-3
      # self.var[g, :, :, :2] = self.var[g, :, :, :2] / uRef
      # self.var[g, :, :,  2] = self.var[g, :, :,  2] / (uRef*uRef)
    # normalize using bounds
    for i in range(3):
      self.bounds[i,:] = np.amin(self.var[:,:,:,i]), np.amax(self.var[:,:,:,i])
      #self.var[:,:,:,i] = 2.0 * (self.var[:,:,:,i] - self.bounds[i,0]) \
                          #/ (self.bounds[i,1] - self.bounds[i,0]) - 1.0


  def summary(self):
    if not self.isLoaded:
      print('NS DataSet has not been loaded yet, end summary')
      return
    print('--------------------------------')
    print('Load data from files:')
    for f in self.fNames:
      print('  {}'.format(f))
    print('Total Genomes {}'.format(self.nTotalGenome))
    print('Genome size {} {}'.format(self.gSize[0], self.gSize[1]))
    print('Range of coordinates:')
    print('  x {} {}'.format(np.amin(self.xy[:,:,:,0]),  np.amax(self.xy[:,:,:,0])))
    print('  y {} {}'.format(np.amin(self.xy[:,:,:,1]),  np.amax(self.xy[:,:,:,1])))
    print('  u {} {}'.format(self.bounds[0,0], self.bounds[0,1]))
    print('  v {} {}'.format(self.bounds[1,0], self.bounds[1,1]))
    print('  p {} {}'.format(self.bounds[2,0], self.bounds[2,1]))
    print('--------------------------------')


  def plot_genome_range(self):
    plt.figure(figsize=(15, 5))
    nBcPoint = 2*(self.gSize[0]+self.gSize[1]-2)
    bcMax, bcMin = np.zeros(3*nBcPoint), np.zeros(3*nBcPoint)
    for i in range(self.nTotalGenome):
      bc = np.concatenate((self.bc[i,:,0], self.bc[i,:,1], self.bc[i,:,2]))
      bcMax, bcMin = np.maximum(bcMax, bc), np.minimum(bcMin, bc)
      plt.plot(np.arange(3*nBcPoint), bc, 'b-', lw=0.1)
    plt.plot(np.arange(3*nBcPoint), bcMax, 'r-')
    plt.plot(np.arange(3*nBcPoint), bcMin, 'g-')


  def num_genome(self):
    return self.nTotalGenome


  def init_collocation_points(self, nCollocPoint, stride=2, nearBoundary=2):
    '''
    uniformly distribute the collocation points across grid cells
    Current version only works with uniform grid
    '''
    # allocate collocation points
    self.nColPnt  = nCollocPoint
    self.xyCol    = np.zeros((self.nTotalGenome, self.nColPnt, 2))
    # set init distribution density based on distance to boundaries
    nSolPerBatch = 20
    nCell        = self.gSize[0] - 1
    gNorm        = np.zeros((nSolPerBatch, nCell, nCell))
    for ss in range(nSolPerBatch):
      for i in range(nCell):
        for j in range(nCell):
          gNorm[ss, i, j] = max(abs(i-nCell/2+0.5), abs(j-nCell/2+0.5))
    for s in range(0, self.nTotalGenome, nSolPerBatch):
      self.adapt_collocation(gNorm, s, stride=stride, nearBoundary=nearBoundary)


  def adapt_collocation(self, gradNorm, solBegin, stride=2,\
                       nearBoundary=None):
    nCell = self.gSize[0] - 1
    assert gradNorm.shape[1] == nCell and gradNorm.shape[2] == nCell

    nSampleCell = (nCell//stride) * (nCell//stride)
    nSol  = gradNorm.shape[0]
    for s in range(solBegin, min(solBegin+nSol, self.nTotalGenome)):
      ss = s - solBegin
      # get number of points in each cell
      nPntInCell  = np.zeros(nSampleCell, dtype=int)
      density = gradNorm[ss,:,:] / np.sum(gradNorm[ss,:,:])
      if nearBoundary != None:
        density = np.power(density, nearBoundary)
        density = density / np.sum(density)
      m = 0
      for i  in range(0, nCell, stride):
        for j in range(0, nCell, stride):
          # number of points in current cell: stride x stride
          cellDensity   = np.sum(density[i:i+stride, j:j+stride])
          nPntInCell[m] = max(int(np.rint(cellDensity * self.nColPnt)), 1)
          m += 1
      # adjust number of points to fit nPoint
      diff = self.nColPnt - np.sum(nPntInCell)
      m    = 0
      if diff > 0:
        while diff > 0:
          nPntInCell[m] += 1
          diff -= 1
          m = (m + 1) % nSampleCell
      else:
        while diff < 0:
          if nPntInCell[m] > 1:
            nPntInCell[m] -= 1
            diff += 1
          m = (m + 1) % nSampleCell
      # generate points
      m = 0
      h = (self.xy[0,0,-1,0] - self.xy[0,0,0,0]) / (self.gSize[0]-1)
      ps, pe = 0, 0
      for i  in range(0, nCell, stride):
        for j in range(0, nCell, stride):
          ps, pe = pe, pe + nPntInCell[m]
          self.xyCol[s, ps:pe, :]  = np.random.rand(nPntInCell[m], 2) * stride * h
          self.xyCol[s, ps:pe, 0] += i * h
          self.xyCol[s, ps:pe, 1] += j * h
          m += 1


  def generator_bcXybcRe_xy_w_label(self, gBegin, gEnd, nGPerBatch, \
                                    nDataPoint1D=16, separateRe=False):
    # set number of boundary, data points, and batch size
    nPnti, nPntj = self.gSize
    nBcPnt       = 2 * (nPnti + nPntj - 2)
    nDatPnt      = nDataPoint1D * nDataPoint1D
    nPntPerBatch = nBcPnt + nDatPnt + self.nColPnt
    batchSize    = nGPerBatch * nPntPerBatch
    # set indices for data points
    datPntIdxi   = np.linspace(1, self.gSize[0]-2, nDataPoint1D, dtype=int)
    datPntIdxj   = np.linspace(1, self.gSize[1]-2, nDataPoint1D, dtype=int)
    # local variables
    gBc          = np.zeros((nBcPnt, 3))
    gXyBc        = np.zeros((nBcPnt, 2))
    # allocate inputs for the netowork
    bcXybcRe     = np.zeros((batchSize, nBcPnt*5+1))
    xy           = np.zeros((batchSize, 2))
    label        = np.zeros((batchSize, 3))
    w            = np.zeros((batchSize, 1))

    # print basic info
    print('--------------------------------')
    print('generator:')
    print('{} boundary points'.format(nBcPnt))
    print('{} data points'.format(nDatPnt))
    print('{} collocation points'.format(self.nColPnt))
    if separateRe:
      print('output [bcXybc, Re, xy, w], label')
    else:
      print('output [bcXybcRe, xy, w], label')
    print('--------------------------------')

    # set bc and label per solution
    gBatchBegin = gBegin
    while 1:
      if gBatchBegin + nGPerBatch > gEnd: gBatchBegin = gBegin
      for g in range(gBatchBegin, gBatchBegin + nGPerBatch):
        gg = g - gBatchBegin
        # ---- boundary points coordinates and values ---- #
        # current genome's bc values and coordinates
        s, e = 0, nPntj-1
        gBc[s:e, :]   = self.var[g,  0, :-1, :]
        gXyBc[s:e, :] = self.xy[g, 0, :-1, :]
        s, e = e, e + nPnti-1,
        gBc[s:e, :]   = self.var[g, :-1, -1, :]
        gXyBc[s:e, :] = self.xy[g, :-1, -1, :]
        s, e = e, e + nPntj-1
        gBc[s:e, :]   = self.var[g, -1, -1:0:-1, :]
        gXyBc[s:e, :] = self.xy[g, -1, -1:0:-1, :]
        s, e = e, e + nPnti-1
        gBc[s:e, :]   = self.var[g, -1:0:-1, 0, :]
        gXyBc[s:e, :] = self.xy[g, -1:0:-1, 0, :]
        # broadcast bc, xyBc, Re to all points in one genome
        for m in range(gg*nPntPerBatch, (gg+1)*nPntPerBatch):
          bcXybcRe[m,           :nBcPnt*3] = gBc[:,:].flatten()
          bcXybcRe[m, nBcPnt*3:nBcPnt*5]   = gXyBc[:,:].flatten()
          bcXybcRe[m, nBcPnt*5]            = self.Re[g]
        # ---- coordinates input and labels---- #
        # boundary points
        iPnt = gg * nPntPerBatch
        xy[iPnt:iPnt+nBcPnt, :]    = gXyBc[...]
        label[iPnt:iPnt+nBcPnt, :] = gBc[:, :]
        # data points
        iPnt += nBcPnt
        for i in datPntIdxi:
          for j in datPntIdxj:
            xy[iPnt, :]    = self.xy[g, i, j, :]
            label[iPnt, :] = self.var[g, i, j, :]
            iPnt += 1
        # collocation points
        xy[iPnt:iPnt+self.nColPnt, :]    = self.xyCol[g, :, :]
        label[iPnt:iPnt+self.nColPnt, :] = 0.0
        # ---- distinguish collocation and others ---- #
        s, e      = gg * nPntPerBatch, gg * nPntPerBatch + nBcPnt + nDatPnt
        w[s:e, 0] = 1.0
        s, e      = e, e + self.nColPnt
        w[s:e, 0] = 0.0
      gBatchBegin += nGPerBatch

      if separateRe:
        yield [bcXybcRe[:,:-1], bcXybcRe[:,-1].reshape(batchSize,1), xy, w], label
      else:
        yield [bcXybcRe, xy, w], label


  def generator_bcXybcRe_xy_label(self, gBegin, gEnd, nGPerBatch, iStride=1, jStride=1, epochShuffle=100000):
    nPnti, nPntj = self.gSize
    nBcPnt  = 2 * (nPnti + nPntj - 2)
    nDatPnt  = ((nPnti-2) // iStride) * ((nPntj-2) // jStride)
    print('--------------------------------')
    print('generator: {} bc points, {} inner points'.format(nBcPnt, nDatPnt))
    print('--------------------------------')
    nPointPerBatch = nBcPnt + nDatPnt
    batchSize = nGPerBatch * nPointPerBatch
    epoch     = 0
    # nBcPnt * (u, v, p, x, y), Re
    bcXybcRe  = np.zeros((batchSize, nBcPnt*5+1))
    gBc       = np.zeros((nBcPnt, 3))
    gXyBc     = np.zeros((nBcPnt, 2))
    xyColloc  = np.zeros((batchSize, 2))
    label     = np.zeros((batchSize, 3))

    # set bc and label per solution
    gBatchBegin = gBegin
    while 1:
      if gBatchBegin + nGPerBatch > gEnd:
        gBatchBegin  = gBegin
        epoch += 1
        if epoch % epochShuffle == 0:
          np.random.shuffle(self.shuffledIndices)
          self.xy[...]  = self.xy[self.shuffledIndices,...]
          self.var[...] = self.var[self.shuffledIndices,...]
          self.loc[...] = self.loc[self.shuffledIndices,...]
          self.Re[...]  = self.Re[self.shuffledIndices]
      for g in range(gBatchBegin, gBatchBegin + nGPerBatch):
        gg = g - gBatchBegin
        # current genome's bc values and coordinates
        s, e = 0, nPntj-1
        gBc[s:e, :]   = self.var[g,  0, :-1, :]
        gXyBc[s:e, :] = self.xy[g, 0, :-1, :]
        s, e = e, e + nPnti-1,
        gBc[s:e, :]   = self.var[g, :-1, -1, :]
        gXyBc[s:e, :] = self.xy[g, :-1, -1, :]
        s, e = e, e + nPntj-1
        gBc[s:e, :]   = self.var[g, -1, -1:0:-1, :]
        gXyBc[s:e, :] = self.xy[g, -1, -1:0:-1, :]
        s, e = e, e + nPnti-1
        gBc[s:e, :]   = self.var[g, -1:0:-1, 0, :]
        gXyBc[s:e, :] = self.xy[g, -1:0:-1, 0, :]
        # broadcast bc to every point
        for m in range(gg*nPointPerBatch, (gg+1)*nPointPerBatch):
          bcXybcRe[m,         :nBcPnt*3] = gBc[:,:].flatten()
          bcXybcRe[m, nBcPnt*3:nBcPnt*5] = gXyBc[:,:].flatten()
          bcXybcRe[m, nBcPnt*5]          = self.Re[g]
        # collocation points' coordinates, including boundary points
        ## boundary points
        s = gg * nPointPerBatch
        xyColloc[s:s+nBcPnt, :] = gXyBc[...]
        ## inner points
        s += nBcPnt
        for i in range(1, nPnti-1, iStride):
          for j in range(1, nPntj-1, jStride):
            xyColloc[s, :] = self.xy[g, i, j, :]
            s += 1
        # label's boundary part
        s = gg * nPointPerBatch
        label[s:s+nBcPnt, :] = gBc[:, :]
        # label's inner part
        s += nBcPnt
        label[s:s+nDatPnt, :] = np.reshape(self.var[g, 1:-1:iStride, 1:-1:jStride, :],\
                                            (nDatPnt, 3), order='C')
      gBatchBegin += nGPerBatch
      yield [bcXybcRe, xyColloc], label


  def generator_bc_xy_label(self, gBegin, gEnd, nGPerBatch, nDataPoint1D=31,\
                            len=0.5, omitP=False):
    nPnti, nPntj = self.gSize
    assert (nPnti == nPntj)
    nBcPnt       = 2 * (nPnti + nPntj - 2)
    nDatPnt      = nDataPoint1D * nDataPoint1D
    nPntPerBatch = nBcPnt + nDatPnt
    batchSize    = nGPerBatch * nPntPerBatch
    # set indices for data points
    datPntIdxi   = np.linspace(1, self.gSize[0]-2, nDataPoint1D, dtype=int)
    datPntIdxj   = np.linspace(1, self.gSize[1]-2, nDataPoint1D, dtype=int)
    # allocate inputs and labels
    xy = np.zeros((batchSize, 2))
    label = np.zeros((batchSize, 3))
    if omitP:
      bc    = np.zeros((batchSize, nBcPnt*2))
    else:
      bc    = np.zeros((batchSize, nBcPnt*3))

    print('--------------------------------')
    print('generator: {} bc points, {} data points'.format(nBcPnt, nDatPnt))
    if omitP: print('Omit pressure')
    print('--------------------------------')

    # set bc and label per solution
    gBatchBegin = gBegin
    while 1:
      if gBatchBegin + nGPerBatch > gEnd:
        gBatchBegin  = gBegin
      for g in range(gBatchBegin, gBatchBegin + nGPerBatch):
        gg = g - gBatchBegin
        # broadcast bc to every point
        if omitP:
          for m in range(gg*nPntPerBatch, (gg+1)*nPntPerBatch):
            bc[m, :] = self.bc[g,:,:2].flatten()
        else:
          for m in range(gg*nPntPerBatch, (gg+1)*nPntPerBatch):
            bc[m, :] = self.bc[g,:,:].flatten()
        # collocation points' coordinates, including boundary points
        ## boundary points
        s = gg * nPntPerBatch
        xy[s:s+nBcPnt, :]    = self.xyBc[g,...]
        label[s:s+nBcPnt, :] = self.bc[g,...]
        ## data points
        s += nBcPnt
        for i in datPntIdxi:
          for j in datPntIdxj:
            xy[s, :]   = self.xy[g, i, j, :] #[j*h, i*h]
            label[s,:] = self.var[g, i, j, :]
            s += 1
      gBatchBegin += nGPerBatch

      yield [bc, xy], label


  def generator_uv_xy_w_label(self, gBegin, gEnd, nGPerBatch, nDataPoint1D=31, len=0.5):
    nPnti, nPntj = self.gSize
    assert (nPnti == nPntj)
    nBcPnt       = 2 * (nPnti + nPntj - 2)
    nDatPnt      = nDataPoint1D * nDataPoint1D
    nPntPerBatch = nBcPnt + nDatPnt + self.nColPnt
    batchSize    = nGPerBatch * nPntPerBatch
    # set indices for data points
    datPntIdxi = np.linspace(1, self.gSize[0]-2, nDataPoint1D, dtype=int)
    datPntIdxj = np.linspace(1, self.gSize[1]-2, nDataPoint1D, dtype=int)
    # nBcPnt * (u, v, p)
    bc    = np.zeros((batchSize, nBcPnt*2))
    xy    = np.zeros((batchSize, 2))
    label = np.zeros((batchSize, 3))
    w     = np.zeros((batchSize, 1))

    # print basic info
    print('--------------------------------')
    print('generator:')
    print('{} boundary points'.format(nBcPnt))
    print('{} data points'.format(nDatPnt))
    print('{} collocation points'.format(self.nColPnt))
    print('--------------------------------')

    # set bc and label per solution
    gBatchBegin = gBegin
    while 1:
      if gBatchBegin + nGPerBatch > gEnd:
        gBatchBegin  = gBegin
      for g in range(gBatchBegin, gBatchBegin + nGPerBatch):
        gg = g - gBatchBegin
        # broadcast bc to every point
        for m in range(gg*nPntPerBatch, (gg+1)*nPntPerBatch):
          bc[m, :] = self.bc[g,:,:2].flatten()
        # ---- coordinates input and labels---- #
        # boundary points
        s = gg * nPntPerBatch
        xy[s:s+nBcPnt, :]    = self.xyBc[g,...]
        label[s:s+nBcPnt, :] = self.bc[g,...]
        # data points
        s += nBcPnt
        if nDatPnt > 0:
          for i in datPntIdxi:
            for j in datPntIdxj:
              xy[s, :]   = self.xy[g, i, j, :] #[j*h, i*h]
              label[s,:] = self.var[g, i, j, :]
              s += 1
        # collocation points
        xy[s:s+self.nColPnt, :]    = self.xyCol[g, :, :]
        label[s:s+self.nColPnt, :] = 0.0
        # ---- distinguish collocation and others ---- #
        s, e      = gg * nPntPerBatch, gg * nPntPerBatch + nBcPnt + nDatPnt
        w[s:e, 0] = 1.0
        s, e      = e, e + self.nColPnt
        w[s:e, 0] = 0.0
      gBatchBegin += nGPerBatch

      yield [bc, xy, w], label


  def generate_genome_input(self, g, nDataPoint1D=31):
    nPnti, nPntj = self.gSize
    assert (nPnti == nPntj)
    nBcPnt    = 2*(nPnti + nPntj - 2)
    nDatPnt   = nDataPoint1D * nDataPoint1D
    bc        = np.zeros((nDatPnt, nBcPnt*2))
    xy        = np.zeros((nDatPnt, 2))
    w         = np.zeros((nDatPnt, 1))
    # set bc
    for i in range(nDatPnt):
      bc[i, :] = self.bc[g, :, :2].flatten()
    # set xy
    datPntIdxi = np.linspace(0, self.gSize[0]-1, nDataPoint1D, dtype=int)
    datPntIdxj = np.linspace(0, self.gSize[1]-1, nDataPoint1D, dtype=int)
    m = 0
    for i in datPntIdxi:
      for j in datPntIdxj:
        xy[m, :] = self.xy[g, i, j, :]
        m += 1
    assert m == nDatPnt
    w[...] = 1.0
    return [bc, xy, w]


  def extract_genome_bc(self):
    nPnti, nPntj = self.gSize[0], self.gSize[1]
    nBcPoint = 2*(self.gSize[0] + self.gSize[1] - 2)
    for g in range(self.nTotalGenome):
      s, e = 0, nPntj-1
      self.bc[g,s:e,:]   = self.var[g,  0, :-1, :]
      self.xyBc[g,s:e,:] = self.xy[g, 0, :-1, :]
      s, e = e, e + nPnti-1,
      self.bc[g,s:e,:]   = self.var[g, :-1, -1, :]
      self.xyBc[g,s:e,:] = self.xy[g, :-1, -1, :]
      s, e = e, e + nPntj-1
      self.bc[g,s:e,:]   = self.var[g, -1, -1:0:-1, :]
      self.xyBc[g,s:e,:] = self.xy[g, -1, -1:0:-1, :]
      s, e = e, e + nPnti-1
      self.bc[g,s:e,:]   = self.var[g, -1:0:-1, 0, :]
      self.xyBc[g,s:e,:] = self.xy[g, -1:0:-1, 0, :]


  def diff_bc(self, bc, shapeID, nGenomePerSolution=100):
    assert self.shuffledIndices == None
    assert shapeID < nGenomePerSolution
    assert len(bc) == self.bc.shape[1] * self.bc.shape[2]

    gStep = nGenomePerSolution
    gid   = shapeID
    errMax = np.zeros((3, 2))
    for i in range(3):
      for g in range(shapeID, nTotalGenome, gStep):
        var0 = bc[i::3]
        var1 = self.bc[g,:,i]
        err  = np.absolute(var0-var1)
        rerr = err / np.absolute(var2+var1) * 2
        err  = np.sum(err)
        if errMax[i,0] < err:
          errMax[i,0] = err
          errMax[i,1] = np.sum(rerr)
    return errMax
