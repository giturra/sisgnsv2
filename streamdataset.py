import numpy as np
import torch as T
device = T.device("cpu")
from tqdm import tqdm


class TweetStreamLoader: 

  def __init__(self, fn, bat_size, buff_size,
      shuffle=False, seed=0):
    
    if buff_size % bat_size != 0:
      raise Exception("buff_size must be evenly div by bat_size")

    self.bat_size = bat_size
    self.buff_size = buff_size 
    self.shuffle = shuffle

    self.rnd = np.random.RandomState(seed)

    self.ptr = 0              # points into x_data and y_data
    self.fin = open(fn, encoding='utf-8')  # line-based text file

    self.buffer = []      
    self.tweets = None        
    self.reload_buffer()      

  def reload_buffer(self):
    self.buffer = []       
    self.ptr = 0
    ct = 0       # number of lines read
    while ct < self.buff_size:
      line = self.fin.readline()
      if line == "":
        self.fin.seek(0)
        return -1  # reached EOF
      else:
        line = line.strip()  # remove trailing newline
        self.buffer.append(line)  
        ct += 1

    if len(self.buffer) != self.buff_size:
      return -2  # buffer was not fully loaded

    if self.shuffle == True:
      self.rnd.shuffle(self.buffer)  # in-place

    return 0  # buffer successfully loaded

  def __iter__(self):
    return self

  def __next__(self):  # next batch as a tuple
    res = 0 

    if self.ptr + self.bat_size > self.buff_size:  # reload
      print(" ** reloading buffer ** ")
      res = self.reload_buffer() 
      # 0 = success, -1 = hit eof, -2 = not fully loaded 

    if res == 0:
      start = self.ptr
      end = self.ptr + self.bat_size
      text = self.buffer[start: end]
      self.ptr += self.bat_size
      return text

    # reached end-of-epoch (EOF), so signal no more
    self.reload_buffer()  # prepare for next epoch
    raise StopIteration
 
# -----------------------------------------------------------

def main():
  print("\nBegin streaming data loader demo ")
  np.random.seed(1)

  fn = "D:/u/tesis/twitterStream-20091110-20100201-v0.1.1/twitterStream-20091110-20100201-v0.1.1"  # 40 lines of data
  bat_size = 8 
  buff_size = 24  # a multiple of bat_size
  emp_ldr = TweetStreamLoader(fn, bat_size, buff_size, \
    shuffle=False) 

  max_epochs = 1
  # for epoch in range(max_epochs):
  #   #print("\n == Epoch: " + str(epoch) + " ==")
  tweets_file = open('100000tweet.txt', "w", encoding='utf-8')
  i = 0
  for (b_idx, batch) in enumerate(emp_ldr):
    #print("epoch: " + str(epoch) + "   batch: " + str(b_idx)) 
    tweet = batch[0].split('\t')[2]  # predictors
    tweets_file.write(f'{tweet}\n')
    if i == 1e6:
      break
    i += 1
  tweets_file.close()
  emp_ldr.fin.close()

print("End demo ")

if __name__ == "__main__":
  main()