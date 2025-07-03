import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

class DataVisualizer:
    """Visualization utilities"""
    
    def plot_length_distribution(self, lengths: list) -> None:
        plt.figure(figsize=(8, 4))
        sns.histplot(lengths, bins=20, kde=True)
        plt.title('Caption Length Distribution')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.show()
    
    def create_wordcloud(self, texts: list, language: str = 'en') -> None:
        text = ' '.join(texts)
        wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'WordCloud ({language})')
        plt.show()
    
    def plot_training_history(self, history: dict) -> None:
        plt.figure(figsize=(10, 4))
        if 'accuracy' in history:
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'], label='Train Acc')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Val Acc')
            plt.title('Accuracy')
            plt.legend()
        if 'loss' in history:
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Loss')
            plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, labels: list) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray, input_tokens: list, output_tokens: list) -> None:
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            x=input_tokens,
            y=output_tokens,
            colorscale='Viridis'))
        fig.update_layout(title='Attention Heatmap', xaxis_title='Input Tokens', yaxis_title='Output Tokens')
        fig.show()