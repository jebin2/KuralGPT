import json
import re

class KuralToken:
    def __init__(self):
        self.kurals = []
        self.kural_vocab = {}

        with open("thirukural.json", "r", encoding="utf-8") as file:
            local_kurals = json.load(file)

        unique_words = set()
        
        for item in local_kurals:
            text = item.get('kural_in_tamil', '')
            text = re.sub(r'[^\u0B80-\u0BFF\s]', '', text)
            self.kurals.append(text)
            words = text.split()

            unique_words.update(words)

        self.kural_vocab = {word: idx for idx, word in enumerate(unique_words)}

def get_token():
    kuralToken = KuralToken()
    return kuralToken.kural_vocab

if __name__ == "__main__":
    print(get_token())
