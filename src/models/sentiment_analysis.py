import ray
import torch
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@ray.remote
class SentimentAnalyzer:
    def __init__(self, train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels):
        self.train_inputs = train_inputs
        self.test_inputs = test_inputs
        self.train_masks = train_masks
        self.test_masks = test_masks
        self.train_labels = train_labels
        self.test_labels = test_labels

    def train_model(self, config):
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False,
        )

        optimizer = AdamW(model.parameters(), lr=config["lr"])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(config["epochs"]):
            model.train()
            optimizer.zero_grad()
            outputs = model(self.train_inputs.to(device), 
                            token_type_ids=None, 
                            attention_mask=self.train_masks.to(device), 
                            labels=self.train_labels.to(device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(self.test_inputs.to(device), 
                            token_type_ids=None, 
                            attention_mask=self.test_masks.to(device))
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        return {
            "accuracy": accuracy_score(self.test_labels, predictions),
            "precision": precision_score(self.test_labels, predictions, average='weighted'),
            "recall": recall_score(self.test_labels, predictions, average='weighted'),
            "f1": f1_score(self.test_labels, predictions, average='weighted')
        }