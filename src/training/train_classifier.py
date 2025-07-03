import pandas as pd
import numpy as np
from src.data_processor import Flickr8kProcessor, TextPreprocessor
from src.models.classification_models import RNNClassifier, LSTMClassifier, AttentionClassifier
from src.utils.metrics import MetricsCalculator
import joblib

# Đường dẫn data
DATA_PATH = '../../data'

# 1. Load và tiền xử lý dữ liệu
processor = Flickr8kProcessor(DATA_PATH)
captions_df = processor.load_captions()
captions_df = processor.create_length_labels(captions_df)

# 2. Tiền xử lý text
text_prep = TextPreprocessor(language='en')
tokenized = text_prep.tokenize(captions_df['caption'].tolist())
vocab = text_prep.build_vocabulary(tokenized, vocab_size=5000)
sequences = text_prep.texts_to_sequences(captions_df['caption'].tolist(), vocab)
padded = text_prep.pad_sequences(sequences, maxlen=20)

# 3. Chuẩn bị label
label_map = {'short': 0, 'medium': 1, 'long': 2}
y = captions_df['length_category'].map(label_map).values

# 4. Chia train/test
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(padded, y, test_size=0.2, random_state=42)

# 5. Train các mô hình
results = {}
metrics = MetricsCalculator()

for ModelClass, name in zip([RNNClassifier, LSTMClassifier, AttentionClassifier], ['RNN', 'LSTM', 'Attention']):
    print(f'\nTraining {name}...')
    model = ModelClass(vocab_size=len(vocab), num_classes=3)
    model.build_model()
    history = model.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64)
    y_pred = np.argmax(model.model.predict(X_val), axis=1)
    result = metrics.calculate_classification_metrics(y_val, y_pred)
    print(f'{name} metrics:', result)
    results[name] = result
    # Lưu model
    model.model.save(f'{name.lower()}_classifier.h5')
    joblib.dump(vocab, f'{name.lower()}_vocab.pkl')

print('\nAll results:', results)
