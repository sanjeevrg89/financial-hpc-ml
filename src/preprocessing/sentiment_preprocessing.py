from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def preprocess_sentiment_data(df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    encoded_data = tokenizer.batch_encode_plus(
        df['text'].tolist(),
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )
    
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = df['sentiment'].values
    
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
    train_masks, test_masks, _, _ = train_test_split(attention_masks, labels, test_size=0.2, random_state=42)
    
    return train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels