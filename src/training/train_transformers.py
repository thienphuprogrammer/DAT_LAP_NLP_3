import pandas as pd
import numpy as np
from src.data_processor import Flickr8kProcessor
from src.models.transformer_models import TransformerClassifier, TransformerTranslator, TransformerCaptioner
from src.utils.metrics import MetricsCalculator
from transformers import AutoTokenizer
import tensorflow as tf

DATA_PATH = '../../data'

# 1. Fine-tune TransformerClassifier cho classification
print('--- Fine-tune TransformerClassifier ---')
processor = Flickr8kProcessor(DATA_PATH)
captions_df = processor.load_captions()
captions_df = processor.create_length_labels(captions_df)
label_map = {'short': 0, 'medium': 1, 'long': 2}
y = captions_df['length_category'].map(label_map).values
texts = captions_df['caption'].tolist()

clf = TransformerClassifier(model_name='distilbert-base-uncased', num_classes=3)
tokenizer = clf.tokenizer
encodings = tokenizer(texts, truncation=True, padding=True, max_length=32)
X = {k: np.array(v) for k, v in encodings.items()}

dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(16)
train_size = int(0.8 * len(texts))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
clf.build_model()
clf.train(train_dataset, val_dataset, epochs=1)
clf.model.save_pretrained('transformer_classifier')
tokenizer.save_pretrained('transformer_classifier')

# 2. Fine-tune TransformerTranslator cho translation
print('--- Fine-tune TransformerTranslator ---')
df = processor.load_translated_captions('vi')
src_texts = df['caption'].tolist()
tgt_texts = df['translated_caption'].tolist()
translator = TransformerTranslator(model_name='Helsinki-NLP/opus-mt-vi-en')
tokenizer = translator.tokenizer
src_enc = tokenizer(src_texts, truncation=True, padding=True, max_length=32, return_tensors='tf')
tgt_enc = tokenizer(tgt_texts, truncation=True, padding=True, max_length=32, return_tensors='tf')
train_dataset = tf.data.Dataset.from_tensor_slices((dict(src_enc), dict(tgt_enc['input_ids']))).batch(8)
translator.build_model()
translator.train(train_dataset, train_dataset, epochs=1)
translator.model.save_pretrained('transformer_translator')
tokenizer.save_pretrained('transformer_translator')

# 3. Fine-tune TransformerCaptioner cho captioning (inference demo)
print('--- Inference TransformerCaptioner ---')
captioner = TransformerCaptioner(model_name='nlpconnect/vit-gpt2-image-captioning')
captioner.build_model()
# Ví dụ inference:
# caption = captioner.generate_caption('path/to/image.jpg')
# print('Generated caption:', caption)
