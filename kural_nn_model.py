import torch
import torch.nn as nn
import torch.nn.functional as F
from kural_model import KuralModel

class KuralNNModel(KuralModel):
	def __init__(self, vocab_size, embedding_dim=64, hidden_layers_dims=None):
		super(KuralNNModel, self).__init__(vocab_size, embedding_dim)

		if hidden_layers_dims is None:
			hidden_layers_dims = [5, 5, 5, 1]  # Include output dimension of 1

		layers = []
		input_dim = embedding_dim
		for i, dims in enumerate(hidden_layers_dims):
			layers.append(nn.Linear(input_dim, dims))
			# Don't add ReLU after the final linear layer
			if i < len(hidden_layers_dims) - 1:
				layers.append(nn.ReLU())
			input_dim = dims

		self.hidden_layers = nn.Sequential(*layers)

	def forward(self, center_words, context_words=None):
		center_embeds = self.in_embeddings(center_words)
		center_hidden = self.hidden_layers(center_embeds)

		context_embeds = self.out_embeddings(context_words)
		context_hidden = self.hidden_layers(context_embeds)

		scores = torch.sum(center_hidden * context_hidden, dim=1)
		return scores
