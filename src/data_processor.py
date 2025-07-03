import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import cv2
import nltk
nltk.download('punkt')

class Flickr8kProcessor:
    """Xử lý dataset Flickr8k"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.captions_file = os.path.join(data_path, 'flickr8k', 'captions.txt')
        self.translated_dir = os.path.join(data_path, 'flickr8k', 'translated_captions')

    def load_captions(self) -> pd.DataFrame:
        data = []
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '\t' in line:
                    image_id, caption = line.split('\t', 1)
                elif ',' in line:
                    image_id, caption = line.split(',', 1)
                else:
                    continue
                data.append({'image_id': image_id, 'caption': caption})
        return pd.DataFrame(data)

    def load_translated_captions(self, language: str) -> pd.DataFrame:
        file_path = os.path.join(self.translated_dir, f'captions_{language}.txt')
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 3:
                    image_id, caption, translated_caption = parts
                    data.append({'image_id': image_id, 'caption': caption, 'translated_caption': translated_caption})
        return pd.DataFrame(data)

    def create_length_labels(self, captions: pd.DataFrame) -> pd.DataFrame:
        def length_category(text):
            n_words = len(nltk.word_tokenize(text))
            if n_words < 5:
                return 'short'
            elif n_words < 10:
                return 'medium'
            else:
                return 'long'
        captions['length_category'] = captions['caption'].apply(length_category)
        return captions

    def train_test_split(self, data: pd.DataFrame, test_size: float = 0.2):
        return train_test_split(data, test_size=test_size, random_state=42)

class TextPreprocessor:
    """Tiền xử lý text"""
    def __init__(self, language: str = 'en'):
        self.language = language
        if language == 'en':
            self.tokenizer = nltk.word_tokenize
        else:
            # Có thể mở rộng cho các ngôn ngữ khác
            self.tokenizer = nltk.word_tokenize

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenizer(self.clean_text(t)) for t in texts]

    def build_vocabulary(self, tokenized_texts: List[List[str]], vocab_size: int = 10000) -> Dict:
        from collections import Counter
        counter = Counter(token for sent in tokenized_texts for token in sent)
        most_common = counter.most_common(vocab_size-2)
        vocab = {word: idx+2 for idx, (word, _) in enumerate(most_common)}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    def texts_to_sequences(self, texts: List[str], vocab: Dict) -> List[List[int]]:
        tokenized = self.tokenize(texts)
        return [[vocab.get(token, vocab['<UNK>']) for token in sent] for sent in tokenized]

    def pad_sequences(self, sequences: List[List[int]], maxlen: int = None) -> np.ndarray:
        return keras_pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

class ImageProcessor:
    """Xử lý ảnh"""
    def __init__(self, model_name: str = 'InceptionV3'):
        if model_name == 'InceptionV3':
            self.model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
            self.preprocess = inception_preprocess
            self.target_size = (299, 299)
        else:
            raise NotImplementedError(f"Model {model_name} not supported.")

    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        features = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_imgs = np.array([self.preprocess_image(p) for p in batch_paths])
            batch_features = self.model.predict(batch_imgs)
            features.append(batch_features)
        return np.vstack(features)

    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = keras_image.load_img(image_path, target_size=self.target_size)
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess(x)
        return x[0]