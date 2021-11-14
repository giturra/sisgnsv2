import time
import torch
import matplotlib.pyplot as plt

from incremental_skipgram import IncrementalSkipGram

from nltk import word_tokenize

from sklearn.manifold import TSNE

from streamdataset import TweetStreamLoader

cuda = torch.device('cuda:0')

fn = '100000tweet.txt'
bat_size = 256
buff_size = 2048

tsl = TweetStreamLoader(fn, bat_size, buff_size)

start = time.time()

with torch.cuda.device(0):
    isn = IncrementalSkipGram(tokenizer=word_tokenize, device=cuda)
    for b_idx, batch in enumerate(tsl):
        isn.fit(batch)
    tsl.close()

end = time.time()

delta_time  = end - start

print(f'Time interval: {delta_time}.')

vocab = list(isn.vocab.table.keys())[0:100]
embeddings = [
    isn.get_embedding(word).detach().numpy() for word in vocab
]

tsne = TSNE(n_components=2)
result = tsne.fit_transform(embeddings)

plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(vocab):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.title('Incremental SGNS 100 Words from Vocabulary')
plt.savefig('grafico_100000_tweet.png')

torch.save(isn.model.state_dict(), './isn_model_100000_tweet.path')