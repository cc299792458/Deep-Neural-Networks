from datasets import load_dataset
from transformers import AutoTokenizer

class Seq2SeqDataset:
    """
    Class to load and tokenize a dataset for Seq2Seq tasks (e.g., machine translation).
    """
    def __init__(self, dataset_name, config_name, source_lang, target_lang, model_name, max_length=256):
        """
        Initialize the Seq2SeqDataset with a specific dataset and tokenizer.

        :param dataset_name: Name of the dataset to load (e.g., 'iwslt2017')
        :param config_name: Configuration of the dataset (e.g., 'iwslt2017-en-zh')
        :param source_lang: Source language key in the dataset (e.g., 'en')
        :param target_lang: Target language key in the dataset (e.g., 'zh')
        :param model_name: Pretrained tokenizer model (e.g., 'Helsinki-NLP/opus-mt-en-zh')
        :param max_length: Maximum tokenization length
        """
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def load_data(self):
        """
        Load and return the dataset.
        """
        dataset = load_dataset(self.dataset_name, self.config_name, trust_remote_code=True)
        return dataset['train'], dataset['validation'], dataset['test']
    
    def tokenize(self, batch):
        """
        Tokenize the source and target languages in the dataset.

        :param batch: Batch of data containing source and target language sentences
        :return: Tokenized input ids, attention masks, and target labels
        """
        source_texts = [example['en'] for example in batch['translation']]
        target_texts = [example['zh'] for example in batch['translation']]
        
        source = self.tokenizer(source_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        target = self.tokenizer(target_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": source['input_ids'], 
            "attention_mask": source['attention_mask'], 
            "labels": target['input_ids']
        }


    def prepare_dataset(self, dataset):
        """
        Apply tokenization to the dataset.

        :param dataset: Raw dataset loaded from `load_data()`
        :return: Tokenized dataset
        """
        return dataset.map(self.tokenize, batched=True)

if __name__ == '__main__':
    dataset_name = 'iwslt2017'
    config_name = 'iwslt2017-en-zh'
    source_lang = 'en'
    target_lang = 'zh'
    model_name = 'Helsinki-NLP/opus-mt-en-zh'
    seq2seq_dataset = Seq2SeqDataset(dataset_name, config_name, source_lang, target_lang, model_name)

    # Load the dataset
    train_data, val_data, test_data = seq2seq_dataset.load_data()

    # Tokenize the dataset
    train_data = seq2seq_dataset.prepare_dataset(train_data)
    val_data = seq2seq_dataset.prepare_dataset(val_data)
    test_data = seq2seq_dataset.prepare_dataset(test_data)