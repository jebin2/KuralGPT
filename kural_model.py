import torch
import torch.nn as nn

class KuralModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim):
		'''
		We use two embeddings because the skip-gram model predicts a context word (target) given a center word (input).
		- `in_embeddings`: Stores embeddings for center words (input).
		- `out_embeddings`: Stores embeddings for context words (output).
		During training, the model learns to make correct (input, target) pairs more similar while reducing similarity with randomly sampled negative words.'
		'''
		super(KuralModel, self).__init__()
		self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)


	def forward(self, center_words, context_words=None):
		center_embeds = self.in_embeddings(center_words)

		context_embeds = self.out_embeddings(context_words)

		scores = torch.sum(center_embeds * context_embeds, dim=1)
		return scores
			
	def get_word_embedding(self, word_idx):
		"""Get the trained embedding for a word"""
		return self.in_embeddings(torch.tensor([word_idx]))
