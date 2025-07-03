import pandas as pd
import numpy as np
from src.data_processor import Flickr8kProcessor, TextPreprocessor
from src.models.translation_models import Seq2SeqTranslator
from src.utils.metrics import MetricsCalculator
import joblib

# Đường dẫn data
DATA_PATH = '../../data'
LANG = 'vi'  # ví dụ: dịch tiếng Việt

# 1. Load và tiền xử lý dữ liệu
processor = Flickr8kProcessor(DATA_PATH)
df = processor.load_translated_captions(LANG)

src_texts = df['caption'].tolist()
tgt_texts = df['translated_caption'].tolist()

src_prep = TextPreprocessor(language='en')
tgt_prep = TextPreprocessor(language=LANG)

src_tokenized = src_prep.tokenize(src_texts)
tgt_tokenized = tgt_prep.tokenize(tgt_texts)

src_vocab = src_prep.build_vocabulary(src_tokenized, vocab_size=5000)
tgt_vocab = tgt_prep.build_vocabulary(tgt_tokenized, vocab_size=5000)

src_seqs = src_prep.texts_to_sequences(src_texts, src_vocab)
tgt_seqs = tgt_prep.texts_to_sequences(tgt_texts, tgt_vocab)

src_padded = src_prep.pad_sequences(src_seqs, maxlen=20)
tgt_padded = tgt_prep.pad_sequences(tgt_seqs, maxlen=20)

# 2. Chia train/test
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(src_padded, tgt_padded, test_size=0.2, random_state=42)

# 3. Build và train model
translator = Seq2SeqTranslator(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab))
translator.build_model()
history = translator.train([X_train, y_train[:, :-1]], y_train[:, 1:], [X_val, y_val[:, :-1]], y_val[:, 1:], epochs=5, batch_size=64)

# 4. Đánh giá BLEU
metrics = MetricsCalculator()
preds = []
for i in range(len(X_val)):
    pred_seq = translator.translate(X_val[i])
    preds.append(' '.join([str(idx) for idx in pred_seq]))
refs = [' '.join([str(idx) for idx in seq]) for seq in y_val[:, 1:]]
bleu = metrics.calculate_bleu(refs, preds)
print('BLEU score:', bleu)

# 5. Lưu model và vocab
translator.model.save('seq2seq_translator.h5')
joblib.dump(src_vocab, 'src_vocab.pkl')
joblib.dump(tgt_vocab, 'tgt_vocab.pkl')
