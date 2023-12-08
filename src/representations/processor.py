from .sae import SAE

class EventProcessor():
  """
  """
  def __init__(self, method="sae", data=None, img_sz=[346, 260],time_windows=[30*10**(-2)]):   
    #Create a SAE instance
    self.method=method
    self.time_windows = time_windows
    self.dataset = data

    self.cols, self.rows= img_sz

    self.SAEs=[]
    for delta in time_windows:
      if method=="simple":
        pass
      elif method=="sae":
        #pass multiple time windows to create different SAE
        self.SAEs.append(SAE(self.cols, self.rows, delta))


  def process_batch(self, batch):
    """
     keep in mind that the batch size can be variable
    """
    if self.method =="sae":
        for i in range(len(self.time_windows)-1):
            current_sae=self.SAEs[i]
            #add logic for the different window times SAE
            for ev in batch:
                timestamp, x, y, polarity= ev
                current_sae.frame[x,y] = {"ts": timestamp, "polarity": polarity}
        return self.SAEs
    elif self.method == "simple":
        print("simple processing method not implemented")
        return None
  
  def load_dataset(self, data):
    """
    Load the dataset for processing.
    :param dataset: The dataset to be processed.
    """
    self.dataset = data