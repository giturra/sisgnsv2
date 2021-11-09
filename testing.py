import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from vocab import Vocabulary
from rand import RandomNum
from unigram_table import UnigramTable
from incremental_skipgram import IncrementalSkipGram
from nltk import word_tokenize


cuda = torch.device('cuda:0')

train_iter = AG_NEWS(split='train')

dataloader = DataLoader(train_iter, batch_size=256, shuffle=False)


with torch.cuda.device(0):


    isn = IncrementalSkipGram(tokenizer=word_tokenize, device=cuda)
    i = 0
    for batch in dataloader:
        isn.fit(batch[1])
        print(f'batch num {i}')
        i += 1
    