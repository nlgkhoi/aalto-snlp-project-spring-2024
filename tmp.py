from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import unicodedata
import re

EOS_token = 1
class TranslationDataset(Dataset):
	def __init__(self, csv_path, dataset_type='train', vocab=None):
		df = pd.read_csv(csv_path)
		self.text, self.labels = zip(*[(text, label) for text, label in zip(df['text'], df['label'])])
		self.dataset_type = dataset_type
		self.tokenizer = get_tokenizer('basic_english')
		self._preprocess(vocab)

	def _preprocess(self, vocab=None):
		# preprocess text
		self.text = [self._preprocess_sentence(text) for text in self.text]

		if vocab is None:
			self.vocab = build_vocab_from_iterator(self._yield_tokens(), specials=["<unk>"])
			self.vocab.set_default_index(self.vocab['<unk>'])
			self.vocab.insert_token('<eos>', EOS_token)  # Insert <eos> token with index 1
		else:
			self.vocab = vocab
		self.vocab_size = len(self.vocab)
	
	def _preprocess_sentence(self, sentence):
		# normalize string
		sentence = normalizeString(sentence)
		return sentence
	
	def _yield_tokens(self):
		for text_sample in self.text:
			yield self.tokenizer(text_sample)

	def __len__(self):
		return len(self.text)

	def __getitem__(self, idx):
		input_seq = text_to_indices(self.tokenizer, self.vocab, self.text[idx])
		label = self.labels[idx]
		return input_seq, label

def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)

def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	return s

def text_to_indices(tokenizer, vocab, text_sample):
	tokens = tokenizer(text_sample)
	indices = [vocab[token] for token in tokens]
	indices.append(EOS_token)
	return torch.tensor(indices, dtype=torch.long).view(-1)

def seq_to_tokens(seq, vocab):
    itos = vocab.get_itos()
    return [itos[idx] for idx in seq]