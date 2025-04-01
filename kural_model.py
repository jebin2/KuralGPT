import torch
import torch.nn as nn
import torch.optim as optim
from kural_token import KuralToken
from kural_relation import get_relation
from kural_dataloader import get_data_loader
import torch.nn.functional as F

class KuralModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim):
		super(KuralModel, self).__init__()
		# We use two embeddings because the skip-gram model predicts a context word (target) given a center word (input).
		# - `in_embeddings`: Stores embeddings for center words (input).
		# - `out_embeddings`: Stores embeddings for context words (output).
		# During training, the model learns to make correct (input, target) pairs more similar while reducing similarity with randomly sampled negative words.
		self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)


	def forward(self, center_words, context_words=None):
		center_embeds = self.in_embeddings(center_words)
		if context_words is not None:
			context_embeds = self.out_embeddings(context_words)
			scores = torch.sum(center_embeds * context_embeds, dim=1)
			print(center_words)
			exit()
			return scores
		else:
			return center_embeds
			
	def get_word_embedding(self, word_idx):
		"""Get the trained embedding for a word"""
		return self.in_embeddings(torch.tensor([word_idx]))


def train_kural_model(kural_model, kural_data_loader, vocab_size, num_epochs=10, learning_rate=0.01):
	optimizer = optim.SGD(kural_model.parameters(), lr=learning_rate)
	criterion = nn.BCEWithLogitsLoss()
	
	for epoch in range(num_epochs):
		total_loss = 0
		for center_words, context_words in kural_data_loader:
			batch_size = center_words.size(0)
			
			# Positive samples: (center_word, actual context_word) pairs.
            # The model should learn that these pairs are related, so we assign a label of 1.
			pos_scores = kural_model(center_words, context_words)
			pos_labels = torch.ones(batch_size, device=center_words.device) # Positive class label of 1.
			pos_loss = criterion(pos_scores, pos_labels)
			print(pos_scores.shape)
			print(pos_labels.shape)
			print(pos_loss.shape)
			
			# Negative samples (random words from vocabulary)
			# Generate 5 negative/random samples for each positive sample
			neg_samples = torch.randint(0, vocab_size, (batch_size * 5,), device=center_words.device)

			# Repeat each center word 5 times (for the 5 negative samples)
			center_words_repeated = center_words.repeat_interleave(5)
			neg_scores = kural_model(center_words_repeated, neg_samples)

			# Negative class labels (0 means the center word and these words should NOT be similar)
			neg_labels = torch.zeros(batch_size * 5, device=center_words.device)
			neg_loss = criterion(neg_scores, neg_labels)
			
			# Total loss is the sum of positive and negative losses
			loss = pos_loss + neg_loss
			
			# Backpropagation
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			total_loss += loss.item()
		
		print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(kural_data_loader)}")

def evaluate_similarity(kural_model, word_idx1, word_idx2):
	# Get embeddings from the input embedding layer
	embedding1 = kural_model.get_word_embedding(word_idx1)
	embedding2 = kural_model.get_word_embedding(word_idx2)
	
	# Normalize embeddings for cosine similarity
	embedding1_norm = F.normalize(embedding1, p=2, dim=1)
	embedding2_norm = F.normalize(embedding2, p=2, dim=1)
	
	# Compute cosine similarity
	cosine_sim = torch.sum(embedding1_norm * embedding2_norm)
	similarity = cosine_sim.item()
	
	print(f"Similarity between words: {similarity}")
	return similarity

def process():
	kuralToken = KuralToken()
	kural_relation = get_relation(kuralToken)
	vocab_size = len(kuralToken.kural_vocab)

	kural_data_loader = get_data_loader(kuralToken, kural_relation)

	kuralModel = KuralModel(vocab_size, embedding_dim=100)

	train_kural_model(kuralModel, kural_data_loader, vocab_size, num_epochs=20, learning_rate=0.01)

	if "அகர" in kuralToken.kural_vocab and "முதல" in kuralToken.kural_vocab:
		evaluate_similarity(kuralModel, kuralToken.kural_vocab["அகர"], kuralToken.kural_vocab["முதல"])

	return kuralModel

if __name__ == "__main__":
	process()