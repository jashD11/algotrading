# essay_scoring.py
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_essay_data(path='essays.csv'):
    """Load and preprocess essay dataset"""
    df = pd.read_csv(path)
    essays = df['text'].values
    scores = df['score'].values
    return train_test_split(essays, scores, test_size=0.2)

def initialize_bert_model():
    """Load pretrained BERT model"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    return tokenizer, model

def tokenize_essays(tokenizer, essays, max_length=512):
    """Tokenize text for BERT input"""
    return tokenizer(
        essays.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )

def train_model(model, train_encodings, train_scores):
    """Fine-tune BERT model"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss)
    
    history = model.fit(
        dict(train_encodings),
        train_scores,
        epochs=3,
        batch_size=8,
        validation_split=0.1
    )
    return model, history

def evaluate_model(model, test_encodings, test_scores):
    """Calculate evaluation metrics"""
    preds = model.predict(dict(test_encodings)).logits
    pearson = np.corrcoef(preds.flatten(), test_scores)[0,1]
    mse = np.mean((preds.flatten() - test_scores)**2)
    return {'Pearson': pearson, 'MSE': mse}

if __name__ == "__main__":
    # Load and split data
    X_train, X_test, y_train, y_test = load_essay_data()
    
    # Initialize model
    tokenizer, model = initialize_bert_model()
    
    # Tokenize data
    train_encodings = tokenize_essays(tokenizer, X_train)
    test_encodings = tokenize_essays(tokenizer, X_test)
    
    # Train and evaluate
    model, history = train_model(model, train_encodings, y_train)
    metrics = evaluate_model(model, test_encodings, y_test)
    
    print(f"\nEvaluation Metrics:")
    print(f"Pearson Correlation: {metrics['Pearson']:.3f}")
    print(f"Mean Squared Error: {metrics['MSE']:.3f}")
