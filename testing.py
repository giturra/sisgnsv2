import numpy as np
import matplotlib.pyplot as plt

import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader

from incremental_skipgram import IncrementalSkipGram

from nltk import word_tokenize

from sklearn.manifold import TSNE

# isn = IncrementalSkipGram(vec_size=5, max_vocab_size=10, window_size=2, tokenizer=word_tokenize, device=torch.device('cpu'))

# with open('mini_corpus.txt', encoding='utf-8') as mini_corpus:
#     for line in mini_corpus:
#         isn.fit([line])

# vocab = isn.vocab
# print(vocab)

cuda = torch.device('cuda:0')

train_iter = AG_NEWS(split='train')

dataloader = DataLoader(train_iter, batch_size=256, shuffle=False)

with torch.cuda.device(0):
    isn = IncrementalSkipGram(tokenizer=word_tokenize, device=cuda)
    for batch in dataloader:
        isn.fit(batch[1])

vocab = list(isn.vocab.table.keys())[0:100]
embeddings = [
    isn.get_embedding(word).detach().numpy() for word in vocab
]

# pca = PCA(n_components=2)
tsne = TSNE(n_components=2)
# result = pca.fit_transform(embeddings)
result = tsne.fit_transform(embeddings)

plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(vocab):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.title('Incremental SGNS 100 Words from Vocabulary')
plt.savefig('grafico_ag_news_train.png')

torch.save(isn.model.state_dict(), './isn_model_ag_news.path')