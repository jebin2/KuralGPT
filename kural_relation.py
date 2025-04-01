from kural_token import KuralToken


class KuralRelation:
	def __init__(self, kuralToken=None, relation_type="skip_gram"):
		self.relation_type = relation_type
		self.kuralToken = kuralToken if kuralToken else KuralToken()
		self.kural_vocab = self.kuralToken.kural_vocab
		self.kurals = self.kuralToken.kurals
		self.relation = []

	def build(self):
		if self.relation_type == "skip_gram":
			self.__skip_gram()

	def __skip_gram(self):
		for item in self.kurals:
			words = item.split()
			word_len = len(words)
			for i, center_word in enumerate(words):
				for j in range(i-2, i + 2):
					if j >= 0 and j != i and j < word_len:
						center_idx = self.kural_vocab[center_word]
						context_idx = self.kural_vocab[words[j]]
						self.relation.append((center_idx, context_idx))

	def get_relation(self):
		return self.relation

def get_relation(kuralToken=None):
	kural_relation = KuralRelation(kuralToken=kuralToken, relation_type="skip_gram")
	kural_relation.build()
	return kural_relation.get_relation()

if __name__ == "__main__":
	print(get_relation())