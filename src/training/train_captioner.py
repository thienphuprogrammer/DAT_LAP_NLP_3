import pandas as pd
import numpy as np
import os
from src.data_processor import Flickr8kProcessor, TextPreprocessor, ImageProcessor
from src.models.captioning_models import ImageCaptioner
from src.utils.metrics import MetricsCalculator
import joblib

# Đường dẫn data
DATA_PATH = '../../data'
IMG_DIR = os.path.join(DATA_PATH, 'flickr8k', 'Images')

# 1. Load và tiền xử lý dữ liệu
processor = Flickr8kProcessor(DATA_PATH)
captions_df = processor.load_captions()

# 2. Tiền xử lý text
text_prep = TextPreprocessor(language='en')
tokenized = text_prep.tokenize(captions_df['caption'].tolist())
vocab = text_prep.build_vocabulary(tokenized, vocab_size=5000)
sequences = text_prep.texts_to_sequences(captions_df['caption'].tolist(), vocab)
padded = text_prep.pad_sequences(sequences, maxlen=20)

# 3. Chuẩn bị ảnh
image_paths = [os.path.join(IMG_DIR, img_id) for img_id in captions_df['image_id']]
img_proc = ImageProcessor(model_name='InceptionV3')
features = img_proc.extract_features(image_paths)

# 4. Chia train/test
from sklearn.model_selection import train_test_split
X_img_train, X_img_val, X_cap_train, X_cap_val = train_test_split(features, padded, test_size=0.2, random_state=42)

# 5. Build và train model
captioner = ImageCaptioner(vocab_size=len(vocab))
captioner.build_model()
history = captioner.train(X_img_train, X_cap_train, epochs=5)

# 6. Đánh giá BLEU
metrics = MetricsCalculator()
preds = []
for i in range(len(X_img_val)):
    pred_seq = captioner.generate_caption(X_img_val[i])
    preds.append(' '.join([str(idx) for idx in pred_seq]))
refs = [' '.join([str(idx) for idx in seq]) for seq in X_cap_val]
bleu = metrics.calculate_bleu(refs, preds)
print('BLEU score:', bleu)

# 7. Lưu model và vocab
captioner.model.save('image_captioner.h5')
joblib.dump(vocab, 'caption_vocab.pkl')
