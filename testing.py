import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from vocab import Vocabulary
from rand import RandomNum
from unigram_table import UnigramTable
from incremental_skipgram import IncrementalSkipGram
from nltk import word_tokenize

train_iter = AG_NEWS(split='train')

dataloader = DataLoader(train_iter, batch_size=8, shuffle=False)

isn = IncrementalSkipGram(tokenizer=word_tokenize)

for batch in dataloader:
    isn.fit(batch[1])
    