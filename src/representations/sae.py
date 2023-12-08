import numpy as np

class SAE:
  def __init__(self, cols, rows, time_window):
    
    self.cols=cols
    self.rows=rows
    self.deltatime = time_window

    self.SAEsize = 5 #what is this parameter for ?

    self.X = np.zeros(self.SAEsize*self.SAEsize, dtype=float)
    self.Y = np.zeros(self.SAEsize*self.SAEsize, dtype=float)
    self.T = np.zeros(self.SAEsize*self.SAEsize, dtype=float)

    self.A = np.zeros((3, 3), dtype=float) # ??

    # self.myveryfirsttime  = 0.0

    self.frame = np.zeros((self.cols, self.rows), dtype = float)
    self.mask = np.zeros((self.SAEsize, self.SAEsize), dtype = bool)

    # self.x = np.zeros(MUCHOS)
    # self.y = np.zeros(MUCHOS)

    # self.validSAEsinthismessage = 0

  
  def clearFrame (self):
    self.frame = np.zeros((self.cols, self.rows), dtype = float)

  def clearMask (self):
    self.mask = np.zeros((self.SAEsize, self.SAEsize), dtype = bool)

  def __del__(self):
    del self.frame
    del self.mask