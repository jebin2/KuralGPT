import torch
from torch.utils.data import DataLoader, TensorDataset
from kural_relation import get_relation
from kural_token import KuralToken

class KuralDataLoader:
    def __init__(self, relation, vocab_size):
        self.relation = relation
        self.vocab_size = vocab_size
        
        # Convert relation to tensors
        self.center_words = torch.tensor([item[0] for item in relation])
        self.context_words = torch.tensor([item[1] for item in relation])

    def get_dataloader(self, batch_size=64):
        dataset = TensorDataset(self.center_words, self.context_words)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_data_loader(kuralToken, relation):
    kuralDataLoader = KuralDataLoader(relation, len(kuralToken.kural_vocab))
    return kuralDataLoader.get_dataloader()

if __name__ == "__main__":
    print(get_data_loader(KuralToken(), get_relation()))