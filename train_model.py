import torch
import torch.nn as nn
import torch.optim as optim
from kural_token import KuralToken
from kural_relation import get_relation
from kural_dataloader import get_data_loader
import torch.nn.functional as F
from kural_model import KuralModel
from kural_nn_model import KuralNNModel

def train_model(model, kural_data_loader, vocab_size, num_epochs=10, learning_rate=0.01):
	model.train()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	criterion = nn.BCEWithLogitsLoss()
	
	for epoch in range(num_epochs):
		total_loss = 0
		for center_words, context_words in kural_data_loader:
			batch_size = center_words.size(0)
			
			# Positive samples: (center_word, actual context_word) pairs.
			# The model should learn that these pairs are related, so we assign a label of 1.
			pos_scores = model(center_words, context_words) # Each one is a sample
			pos_labels = torch.ones(batch_size, device=center_words.device) # Positive class label of 1.
			pos_loss = criterion(pos_scores, pos_labels)
			
			# Negative samples (random words from vocabulary)
			# Generate 5 negative/random samples for each positive sample
			neg_samples = torch.randint(0, vocab_size, (batch_size * 5,), device=center_words.device)

			# Repeat each center word 5 times (for the 5 negative samples)
			center_words_repeated = center_words.repeat_interleave(5)
			neg_scores = model(center_words_repeated, neg_samples)

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

def evaluate_similarity(model, word_idx1, word_idx2):
	model.eval()
	
	with torch.no_grad():
		# Get embeddings from the input embedding layer
		embedding1 = model.get_word_embedding(word_idx1)
		embedding2 = model.get_word_embedding(word_idx2)
		
		# Normalize embeddings for cosine similarity
		embedding1_norm = F.normalize(embedding1, p=2, dim=1)
		embedding2_norm = F.normalize(embedding2, p=2, dim=1)
		print(embedding1)
		# Compute cosine similarity
		cosine_sim = torch.sum(embedding1_norm * embedding2_norm)
		similarity = cosine_sim.item()
		
		print(f"Similarity between words: {similarity}")
		return similarity

def process(type="KuralModel"):
	kuralToken = KuralToken()
	kural_relation = get_relation(kuralToken)
	vocab_size = len(kuralToken.kural_vocab)

	kural_data_loader = get_data_loader(kural_relation)

	if type == "KuralModel":
		model = KuralModel(vocab_size, embedding_dim=100)
	else:
		model = KuralNNModel(vocab_size, embedding_dim=2000)

	train_model(model, kural_data_loader, vocab_size, num_epochs=10, learning_rate=0.01)

	if "அகர" in kuralToken.kural_vocab and "முதல" in kuralToken.kural_vocab:
		evaluate_similarity(model, kuralToken.kural_vocab["அகர"], kuralToken.kural_vocab["முதல"])

	return model

if __name__ == "__main__":
	# process("KuralModel")
	process("KuralNNModel")