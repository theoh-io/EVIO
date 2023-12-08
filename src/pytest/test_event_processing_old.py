# Plot the optical flow based on the SAEs faster than before. Corrected and improved. 
# 2 methods for plane fitting (the second one doesn't work properly)
# The summations to compute of terms needed to obtain full flow are implemented with numpy.sum() -> extremely faster than for loops
# Normal flow and sparse or dense full flow computed.
  
import cv2 # OpenCV library
#----------------------------------------------
import math
from scipy.linalg import svd
from time import time
import colorsys
#----------------------------------------------
import numpy as np
from src.options.config_parser import ConfigParser

PI = 3.1415927

MVSEC = 0 # when using MVSEC dataset
CUBE = 1 # when using cube_regular dataset

COLS_CUBE = 240
ROWS_CUBE = 180

COLS_MVSEC = 346
ROWS_MVSEC = 260



CHAN = 2
SAESIZE = 5
APERTURE_THLD_SAE = 10**(9) # Ratio between evals in SAE problem, the solutions must be well conditioned (rank 2) to have sufficient edge support in both plane directions
MINHITS = 5 # Minimum number of hits in SAE to track an edge
DELTA_TIME = 10**(-2) # Maximum allowed time within SAE in seconds (10ms)
MUCHOS = 300000 # An event arrives each microsec during 30ms, maximum num of events in a message

MAXV_FF = 10 # Maximum module velocity so the image isn't white (normalization of the vel module for the representation), full flow
MAXV_NF = 20 # Maximum module velocity so the image isn't white (normalization of the vel module for the representation), normal flow

FF_WINDOW_SIZE = 35 # Full flow window size
HALF_FF_WINDOW_SIZE = int(FF_WINDOW_SIZE/2)

PLOT_NF = 1 # 1 if plot normal flow
PLOT_FF = 1 # 1 if plot full flow

PLANE_FITTING_METHOD = 1 # 1 for method 1 (SVD), 2 for method 2 (simplest way)

FULL_FLOW = 1 # 1 to compute full flow
SPARSE = 1 # Compute sparse full flow (not 1 when DENSE = 1)
DENSE = 0 # Compute dense full flow (not 1 when SPARSE = 1)




class SAE:
  def __init__(self):
    
    self.cols, self.rows

    self.channels = CHAN
    self.SAEsize = SAESIZE
    self.halfSAEsize = int(SAESIZE/2)
    self.minhits = MINHITS
    self.deltatime = DELTA_TIME

    self.X = np.zeros(self.SAEsize*self.SAEsize, dtype=float)
    self.Y = np.zeros(self.SAEsize*self.SAEsize, dtype=float)
    self.T = np.zeros(self.SAEsize*self.SAEsize, dtype=float)

    self.A = np.zeros((3, 3), dtype=float)

    self.myveryfirsttime  = 0.0

    self.frame = np.zeros((self.cols, self.rows), dtype = float)
    self.mask = np.zeros((self.SAEsize, self.SAEsize), dtype = bool)

    self.velx = np.zeros(MUCHOS)
    self.vely = np.zeros(MUCHOS)
    self.vel = np.zeros((self.cols,self.rows, 2))
    self.x = np.zeros(MUCHOS)
    self.y = np.zeros(MUCHOS)

    self.validSAEsinthismessage = 0

    # Vectors to save the parameters needed to compute the full flow
    self.B00 = np.zeros((self.cols,self.rows))
    self.B11 = np.zeros((self.cols,self.rows))
    self.B10 = np.zeros((self.cols,self.rows))
  
  def clearFrame (self):
    self.frame = np.zeros((self.cols, self.rows), dtype = float)

  def clearMask (self):
    self.mask = np.zeros((self.SAEsize, self.SAEsize), dtype = bool)

  def clearBparam (self):
    self.B00 = np.zeros((self.cols,self.rows))
    self.B11 = np.zeros((self.cols,self.rows))
    self.B10 = np.zeros((self.cols,self.rows))

  def __del__(self):
    del self.frame
    del self.mask

#Create a SAE instance
mySAE = SAE()

class Stats:
  def __init__(self):
    self.time = 0.0
    self.events = 0
    self.frames = 0
    self.validSAEs = 0
    self.eventsoutsideSAE = 0
    self.apertureSAE = 0
    self.noeventsforSAE = 0
    self.eventsInPixel = np.zeros((mySAE.cols, mySAE.rows))
    self.overwrittenPixels = 0
    self.pixelsWithNewEvent = 0
    self.buildFrameTime = 0.0
    self.buildSAEtime = 0.0
    self.checkIfSAE  = 0.0
    self.computeNormalFlowTime = 0.0
    self.computeFullFlowTime = 0.0
    self.computePrevAtime = 0.0
    self.computeAtime = 0.0

  def clearEventsInPixel(self):
    self.eventsInPixel= np.zeros((mySAE.cols, mySAE.rows))

# Create a Stats instance
myStats = Stats()

class EventProcessor():
  """
  Create an EventSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):   
    # Array size computation
    dfsize = len(data.events)
  
    # Update stats
    myStats.frames += 1 # Add new frame to stats
    myStats.events += arraysize # Add new events in the frame to stats
    myStats.validSAEs = 0
    myStats.clearEventsInPixel()
    myStats.overwrittenPixels = 0
    myStats.pixelsWithNewEvent = 0

    # Initialize parameters for this frame
    mySAE.velx = np.zeros(MUCHOS)
    mySAE.vely = np.zeros(MUCHOS)
    mySAE.vel = np.zeros((mySAE.cols, mySAE.rows, 2))
    mySAE.x = np.zeros(MUCHOS)
    mySAE.y = np.zeros(MUCHOS)
    mySAE.validSAEsinthismessage = 0
    mySAE.clearFrame()
    mySAE.clearBparam()
    
    aperture = 0
    total = 0

    start_time_frame = time()
    # For each event in the frame
    for i in range(0, arraysize):
      # Save values x, y for this event
      x = data.events[i].x
      y = data.events[i].y

      # Count the number of events ocurred in pixel (x,y)
      myStats.eventsInPixel[x,y] += 1

      if (i == 0):
        # Start timestamp
        start_t = data.events[i].ts

        if (mySAE.myveryfirsttime == 0.0):
          # Save the value of the first event timestamp
          mySAE.myveryfirsttime = start_t.sec + start_t.nanosec/(10**9)

      if (i == arraysize-1):
        # Save the value of the last event timestamp
        end_t = data.events[i].ts
        
      # Save timestamp in SAE in nanosec
      mySAE.frame[x,y] = data.events[i].ts.sec + data.events[i].ts.nanosec/(10**9)

      #print(mySAE.frame[x][y][int(polarity)])
      #print(data.events[i].ts.nanosec)
      #print(f"Sec: {data.events[i].ts.sec}\nNanosec: {data.events[i].ts.nanosec}\nSum: {data.events[i].ts.sec + data.events[i].ts.nanosec/(10**9)}")
    
    # Compute the time to build the timestamp frame
    myStats.buildFrameTime = time() - start_time_frame
    
    # Inicialize times of the next loop to zero
    myStats.checkIfSAE = 0.0
    myStats.computeNormalFlowTime = 0.0
    myStats.computePrevAtime = 0.0
    myStats.computeAtime = 0.0

    # Save the time we entered in the loop
    start_time_SAEs = time()

    #Pass through every pixel in the image, build SAE and compute its normal flow
    for x in range (0, mySAE.cols):
      for y in range(0, mySAE.rows):
        #print(f"Frame: {mySAE.frame[x][y]}")

        # Check if an event ocurred in the pixel in the message
        if myStats.eventsInPixel[x,y] > 0.0:
          myStats.pixelsWithNewEvent += 1

          #print(f"Dif0: {mySAE.frame[x][y]}")

          # Check if the pixel has been overwritten
          if (myStats.eventsInPixel[x,y] > 1.0):
            myStats.overwrittenPixels += 1

          # Save the time we started to check if the event in the pixel suits the SAE conditions
          start_time_checkIfSAE = time()

          # Check if event does not ocurr in borders of image
          if ((x > mySAE.halfSAEsize) and (x < (mySAE.cols - mySAE.halfSAEsize)) and (y > mySAE.halfSAEsize) and (y < (mySAE.rows - mySAE.halfSAEsize))):
            count = 0 # Valid timestamp within SAE
            mySAE.clearMask()

            # Identify events in SAE close in time to current one
            for u in range (x-mySAE.halfSAEsize, x+mySAE.halfSAEsize):
              for v in range (y-mySAE.halfSAEsize, y+mySAE.halfSAEsize):
                if ((mySAE.frame[x,y] - mySAE.frame[u,v]) < mySAE.deltatime):
                  mySAE.mask[u-x+mySAE.halfSAEsize,v-y+mySAE.halfSAEsize] = 1
                  # Update valid timestamp in SAE
                  count += 1

            #Increment the time checking if the events suit the SAE conditions
            myStats.checkIfSAE += time() - start_time_checkIfSAE
            #print(count)

            # Assuming that you have an edge passing through the SAE you would expect at least
            # 'minhits' events firing, to estimate a plane.
            # Try with a smaller number to let more events produce flow, at the expense of poorly
            #  estimated flow normals

            if (count > mySAE.minhits):
              # Save the time we start to compute the flow
              start_time_computeNormalFlow = time()
              start_time_computePrevA = time()
              
              if PLANE_FITTING_METHOD == 1:              
                # Fitting planes to the events in the SAE
                mux = 0
                muy = 0
                mut = 0

                #print(count)
                for u in range (0, mySAE.SAEsize):
                  x_pos =  x - mySAE.halfSAEsize + u

                  for v in range (0, mySAE.SAEsize):  
                    index_tmp = mySAE.SAEsize*u + v 
                    if (mySAE.mask[u,v]):
                      #print(f"\n(x,y): ({x},{y})\n({x-int(mySAE.SAEsize/2)+u},{y-int(mySAE.SAEsize/2)+v})")
                      #print(f"Iteration: {mySAE.SAEsize*u+v}\nAny: {mySAE.frame[x - int(mySAE.SAEsize/2) + u][y - int(mySAE.SAEsize/2) + v][int(polarity)]}\nCentral: {mySAE.frame[x][y][int(polarity)]}\nDif: {mySAE.frame[x - int(mySAE.SAEsize/2) + u][y - int(mySAE.SAEsize/2) + v][int(polarity)] - mySAE.frame[x][y][int(polarity)]}") 
                      y_pos = y - mySAE.halfSAEsize + v

                      mySAE.X[index_tmp] = x_pos  
                      mySAE.Y[index_tmp] = y_pos
                      mySAE.T[index_tmp] = mySAE.frame[x_pos,y_pos] - mySAE.frame[x,y]
                                  
                    else:
                      mySAE.X[index_tmp] = 0
                      mySAE.Y[index_tmp] = 0
                      mySAE.T[index_tmp] = 0
                      
                #print(f"T: {mySAE.T}")
                # Compute the centroid of the plane
                mux = np.sum(mySAE.X)/count
                muy = np.sum(mySAE.Y)/count
                mut = np.sum(mySAE.T)/count
                
                # print(f"Shape X: {np.shape(mySAE.X)}")
                # print(f"(mux,mux2): ({mux}, {mux2})\n(muy,muy2): ({muy}, {muy2})\n(mut,mut2): ({mut}, {mut2})\n")

                # Increment time for previous computations to build A
                myStats.computePrevAtime += time()-start_time_computePrevA

                # Start time to build A and compute the normal vectors
                start_time_computeA = time()
                #print(f"(x,y): ({x}, {y})\nmux: {mux}\nmuy: {muy}\nmut: {mut}\n")

                # Initialize the 3x3 matrix to solve the LMS problem
                mySAE.A[0,0] = np.dot(mySAE.X, mySAE.X - mux)
                mySAE.A[0,1] = np.dot(mySAE.Y, mySAE.X - mux)
                mySAE.A[0,2] = np.dot(mySAE.T, mySAE.X - mux)
                mySAE.A[1,0] = np.dot(mySAE.X, mySAE.Y - muy)
                mySAE.A[1,1] = np.dot(mySAE.Y, mySAE.Y - muy)
                mySAE.A[1,2] = np.dot(mySAE.T, mySAE.Y - muy)
                mySAE.A[2,0] = np.dot(mySAE.X, mySAE.T - mut)
                mySAE.A[2,1] = np.dot(mySAE.Y, mySAE.T - mut)
                mySAE.A[2,2] = np.dot(mySAE.T, mySAE.T - mut) 

                # Compute Singular Value Decomposition of A. The singular values are in decreasing order.
                U, S, Vt = svd(mySAE.A, full_matrices=True)
                #print(f"\nA: {mySAE.A}\nU: {U}\nS: {S}\nVt: {Vt}")
                #print(f"Vt: {Vt}")

                # Compute velocities in axis x and y
                # Third row in Vt contains the normal vector of the best fit (smaller singular value)
                # We need to scale to project the vector to the x-y plane
                if (Vt[2,0] != 0.0 or Vt[2,1] != 0.0):
                  vx = -Vt[2,2]*Vt[2,0]/(Vt[2,0]**2+Vt[2,1]**2)
                  vy = -Vt[2,2]*Vt[2,1]/(Vt[2,0]**2+Vt[2,1]**2)                

                else:  
                  vx = 0.0
                  vy = 0.0

                #print(f"vx: {vx}\nvy: {vy}\n")
                #print(f"Aperture: {abs(S[0] / S[1])}")
                #print(f"Total: {total}\nAperture: {aperture}")
                total += 1
                # Check the result of SVD is well conditioned. S0/Sn is the condition number. The larger
                # the condition number the more practically non-invertible it is.
                #print(abs(S[2]))
                if (abs(S[0]/S[2]) < APERTURE_THLD_SAE):
                  aperture += 1

                  # Save valid SAEs in the frame
                  mySAE.x[mySAE.validSAEsinthismessage] = x
                  mySAE.y[mySAE.validSAEsinthismessage] = y
                  mySAE.velx[mySAE.validSAEsinthismessage] = vx
                  mySAE.vely[mySAE.validSAEsinthismessage] = vy

                  mySAE.vel[x,y,0] = vx
                  mySAE.vel[x,y,1] = vy
                  
                  mySAE.validSAEsinthismessage += 1 # Update num of valid SAEs in the frame
                  myStats.validSAEs += 1

                  # Save the needed parameters to compute full flow
                  nfsq = np.linalg.norm(np.array([vx,vy]))**2
                  mySAE.B00[x,y] = vy**2/nfsq
                  mySAE.B10[x,y] = vx*vy/nfsq
                  mySAE.B11[x,y] = vx**2/nfsq

                myStats.computeAtime += time()-start_time_computeA
                myStats.computeNormalFlowTime += time()-start_time_computeNormalFlow
              
              if PLANE_FITTING_METHOD == 2:
                # Save the time we start to compute the flow
                start_time_computeNormalFlow = time()

                # Compute the centroid and save points in the SAE to fit the plane
                cx = 0
                cy = 0
                ct = 0

                for u in range(0,mySAE.SAEsize):
                  x_pos = x - mySAE.halfSAEsize + u
                  
                  for v in range (0,mySAE.SAEsize):
                    index_tmp = u*mySAE.SAEsize + v
                    if mySAE.mask[u,v] == 1:
                      y_pos = y - mySAE.halfSAEsize + v
                      
                      cx += x_pos
                      mySAE.X[index_tmp] = x_pos
                      cy += y_pos
                      mySAE.Y[index_tmp] = y_pos
                      ct += mySAE.frame[x_pos,y_pos] - mySAE.frame[x,y]
                      mySAE.T[index_tmp] = mySAE.frame[x_pos,y_pos] - mySAE.frame[x,y]
                    
                    else:
                      mySAE.X[index_tmp] = 0
                      mySAE.Y[index_tmp] = 0
                      mySAE.T[index_tmp] = 0

                cx /= count
                cy /= count
                ct /= count

                # Define the points to be relative to the centroid
                mx = mySAE.X - cx
                my = mySAE.Y - cy
                mt = mySAE.T - ct

                # Compute full 3x3 matrix, excluding symmetries
                mxx = np.dot(mx, mx.transpose())
                mxy = np.dot(mx, my.transpose())
                mxt = np.dot(mx, mt.transpose())
                myy = np.dot(my, my.transpose())
                myt = np.dot(my, mt.transpose())
                mtt = np.dot(mt, mt.transpose())

                # Pick the plane with the best conditioning
                det_x = myy*mtt - myt**2
                det_y = mxx*mtt - mxt**2
                det_t = mxx*myy - mxy**2

                det_max = np.max(np.array([det_x,det_y,det_t]))

                if det_max > 0.0: # The points span a plane
                  if det_max == det_x:
                    n = np.array([det_x, mxt*myt - mxy*mtt, mxy*mxt - mxt*myy])

                  elif det_max == det_y:
                    n = np.array([mxt*myt - mxy*mtt, det_y, mxy*mxt - myt*mxx])
                  
                  else:
                    n = np.array([mxy*myt - mxt*myy, mxy*mxt - myt*mxx, det_t])

                  if (n[0] != 0.0 or n[1] != 0.0):
                    vx = -n[2]*n[0]/(n[0]**2+n[1]**2)
                    vy = -n[2]*n[1]/(n[0]**2+n[1]**2)                

                  else:  
                    vx = 0.0
                    vy = 0.0

                  mySAE.x[mySAE.validSAEsinthismessage] = x
                  mySAE.y[mySAE.validSAEsinthismessage] = y
                  mySAE.velx[mySAE.validSAEsinthismessage] = vx
                  mySAE.vely[mySAE.validSAEsinthismessage] = vy

                  mySAE.validSAEsinthismessage += 1 # Update num of valid SAEs in the frame
                  myStats.validSAEs += 1

                  # Save the needed parameters to compute full flow
                  nfsq = np.linalg.norm(np.array([vx,vy]))**2
                  mySAE.B00[x,y] = vy**2/nfsq
                  mySAE.B10[x,y] = vx*vy/nfsq
                  mySAE.B11[x,y] = vx**2/nfsq

                myStats.computeNormalFlowTime += time()-start_time_computeNormalFlow

    myStats.buildSAEtime = time() - start_time_SAEs

    if FULL_FLOW == 1:
      start_time_computeFullFlow = time()

      if SPARSE == 1:
        for i in range(0, mySAE.validSAEsinthismessage):
          x = int(mySAE.x[i])
          y = int(mySAE.y[i])

          # Define the maximum positions we move in each direction
          lowx_th = x-HALF_FF_WINDOW_SIZE
          lowy_th = y-HALF_FF_WINDOW_SIZE
          highx_th = x+HALF_FF_WINDOW_SIZE
          highy_th = y+HALF_FF_WINDOW_SIZE

          # Check if the pixel is in the borders
          if not((x > HALF_FF_WINDOW_SIZE) and (x < (mySAE.cols - HALF_FF_WINDOW_SIZE)) and (y > HALF_FF_WINDOW_SIZE) and (y < (mySAE.rows - HALF_FF_WINDOW_SIZE))):
            # If the pixel is in the borders, use a non-squared window
            if (x < HALF_FF_WINDOW_SIZE):
              lowx_th = 0
            
            if (x > mySAE.cols - HALF_FF_WINDOW_SIZE):
              highx_th = mySAE.cols

            if (y < HALF_FF_WINDOW_SIZE):
              lowy_th = 0
              
            if (y > mySAE.rows - HALF_FF_WINDOW_SIZE):
              highy_th = mySAE.rows

          # Bv = b -> v = B^(-1)b
          B = np.zeros((2,2))
          B[0,0] = FF_WINDOW_SIZE**2
          B[1][1] = B[0][0]
          b = np.zeros((2,1))

          # Compute the B matrix and b vector.
          B[0,0] -= np.sum(mySAE.B00[lowx_th:highx_th,lowy_th:highy_th])
          B[1,1] -= np.sum(mySAE.B00[lowx_th:highx_th,lowy_th:highy_th])
          B[1,0] = np.sum(mySAE.B10[lowx_th:highx_th,lowy_th:highy_th])
          B[0,1] = B[1,0]

          #print(f"Size: {np.shape(mySAE.vel[x-HALF_FF_WINDOW_SIZE:x+HALF_FF_WINDOW_SIZE,y-HALF_FF_WINDOW_SIZE:y+HALF_FF_WINDOW_SIZE])}")
          vx_sum = np.sum(mySAE.vel[lowx_th:highx_th,lowy_th:highy_th,0])
          vy_sum = np.sum(mySAE.vel[lowx_th:highx_th,lowy_th:highy_th,1])
          b[0] = vx_sum
          b[1] = vy_sum

          # Compute the full flow vector and overwrite the normal flow with the full flow
          ff_vel = np.dot(np.linalg.inv(B),b)
          mySAE.vel[x,y,0] = ff_vel[0]
          mySAE.vel[x,y,1] = ff_vel[1]

      if DENSE == 1:
        for x in range(0, mySAE.cols):
          for y in range(0, mySAE.rows):
            # Define the maximum positions we move in each direction
            lowx_th = x-HALF_FF_WINDOW_SIZE
            lowy_th = y-HALF_FF_WINDOW_SIZE
            highx_th = x+HALF_FF_WINDOW_SIZE
            highy_th = y+HALF_FF_WINDOW_SIZE

            # Check if the pixel is in the borders
            if not((x > HALF_FF_WINDOW_SIZE) and (x < (mySAE.cols - HALF_FF_WINDOW_SIZE)) and (y > HALF_FF_WINDOW_SIZE) and (y < (mySAE.rows - HALF_FF_WINDOW_SIZE))):
              # If the pixel is in the borders, use a non-squared window
              if (x < HALF_FF_WINDOW_SIZE):
                lowx_th = 0
              
              if (x > mySAE.cols - HALF_FF_WINDOW_SIZE):
                highx_th = mySAE.cols

              if (y < HALF_FF_WINDOW_SIZE):
                lowy_th = 0
              
              if (y > mySAE.rows - HALF_FF_WINDOW_SIZE):
                highy_th = mySAE.rows
              
            # Bv = b -> v = B^(-1)b
            B = np.zeros((2,2))
            B[0,0] = FF_WINDOW_SIZE**2
            B[1,1] = B[0,0]
            b = np.zeros((2,1))

            # Compute the B matrix and b vector.
            B[0,0] -= np.sum(mySAE.B00[lowx_th:highx_th,lowy_th:highy_th])
            B[1,1] -= np.sum(mySAE.B00[lowx_th:highx_th,lowy_th:highy_th])
            B[1,0] = np.sum(mySAE.B10[lowx_th:highx_th,lowy_th:highy_th])
            B[0,1] = B[1,0]

            #print(f"Size: {np.shape(mySAE.vel[x-HALF_FF_WINDOW_SIZE:x+HALF_FF_WINDOW_SIZE,y-HALF_FF_WINDOW_SIZE:y+HALF_FF_WINDOW_SIZE])}")
            vx_sum = np.sum(mySAE.vel[lowx_th:highx_th,lowy_th:highy_th,0])
            vy_sum = np.sum(mySAE.vel[lowx_th:highx_th,lowy_th:highy_th,1])
            b[0] = vx_sum
            b[1] = vy_sum

            # Compute the full flow vector and overwrite the normal flow with the full flow
            ff_vel = np.dot(np.linalg.inv(B),b)
            mySAE.vel[x,y,0] = ff_vel[0]
            mySAE.vel[x,y,1] = ff_vel[1]


      myStats.computeFullFlowTime = time()-start_time_computeFullFlow              
        

    # Initial and end time in float type
    tini = start_t.sec+(start_t.nanosec/10**9)
    tend = end_t.sec+(end_t.nanosec/10**9)
      
    # Time between each frame
    T = (end_t.sec+(end_t.nanosec/10**9))-tini

    # Increment the total time
    myStats.time += T

    # Compute the time per SAE
    if myStats.validSAEs > 0:
      timePerSAE = myStats.buildSAEtime/myStats.validSAEs
    
    else:
      timePerSAE = "No valid SAEs"
    

    #Print the stats of the frame
    print("\n----------------------------------------------------------------------")
    print(f"Stats of Frame {myStats.frames}")
    print("----------------------------------------------------------------------")
    print(f"Time: {myStats.time}\nFrom: {tini} to {tend}\nFrame time: {T}\nProcessing time: {myStats.buildFrameTime+myStats.buildSAEtime+myStats.computeFullFlowTime}\n  Time to build frame: {myStats.buildFrameTime}")
    print(f"  Time to build SAEs: {myStats.buildSAEtime}\n    Time to check if SAE: {myStats.checkIfSAE}\n    Time to compute normal flow: {myStats.computeNormalFlowTime}")
    print(f"  Time to compute full flow: {myStats.computeFullFlowTime}")
    #print(f"      Time to compute prev A: {myStats.computePrevAtime}\n      Time to compute A and vector: {myStats.computeAtime}")
    print(f"Events in message: {arraysize}\nTime to build frame per event: {myStats.buildFrameTime/arraysize}")
    print(f"Valid SAEs: {myStats.validSAEs}\nTime per SAE: {timePerSAE}\nOverwritten pixels: {myStats.overwrittenPixels}")
    print(f"  Percentage of overwritten pixels: {myStats.overwrittenPixels/(mySAE.rows*mySAE.cols)*100}%\nPixels with new event: {myStats.pixelsWithNewEvent}\n  Percentage of overwritten pixels with new event: {myStats.overwrittenPixels/myStats.pixelsWithNewEvent*100}%")
    print(f"Events: {myStats.events}\nMax events in pixels: {np.max(myStats.eventsInPixel[:,:])}")
    print(f"Maximum vx full flow: {np.max(mySAE.vel[:,:,0])}\nMaximum vy full flow: {np.max(mySAE.vel[:,:,1])}")

 
def main(args=None):
  
  # Create the node
  event_subscriber = EventSubscriber()
  image_subscriber = ImageSubscriber()

  
if __name__ == '__main__':
  main()