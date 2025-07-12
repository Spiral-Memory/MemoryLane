import json
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from nltk.stem import SnowballStemmer
from sklearn.metrics import accuracy_score
import os

base_dir = os.path.dirname(__file__)
intent_json_path = os.path.join(base_dir, 'intents.json')

with open(intent_json_path) as file:
    data = json.load(file)

patterns = []
labels = []


tag_dict = {}
counter = 0


for sample in data['samples']:
    tag = sample['tags']
    if tag not in tag_dict:
        tag_dict[tag] = counter
        counter += 1
    tag_label = tag_dict[tag]

    for pattern in sample['patterns']:
        patterns.append(pattern)
        labels.append(tag_label)

patterns = np.array(patterns)
labels = np.array(labels)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Initialize K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=321)

class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).clone().detach().long()
        return item

    def __len__(self):
        return len(self.labels)


model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

ignore_words = ['?', '!', '.', ',','is','are','am','was','were','do','does','did','can','could','may','might','must','shall','should','will','would','have','has','had','a','an','the','of','in','on','at','to','for','from','by','with','and','or','but','if','then','else','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
stemmer = SnowballStemmer('english')

# Preprocess all patterns
processed_patterns = []
for pattern in patterns:
    words = pattern.lower().split()
    filtered_words = [word for word in words if word not in ignore_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    processed_patterns.append(' '.join(stemmed_words))

processed_patterns = np.array(processed_patterns)

# Initialize lists to store fold results
fold_accuracies = []
best_model_checkpoint = None
best_accuracy = 0.0

# Perform K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(processed_patterns)):
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1}/5")
    print(f"{'='*50}")
    
    # Split data for this fold
    train_patterns = processed_patterns[train_idx]
    val_patterns = processed_patterns[val_idx]
    train_labels_fold = labels[train_idx]
    val_labels_fold = labels[val_idx]
    
    # Tokenize data for this fold separately
    train_encodings = tokenizer(
        train_patterns.tolist(), truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(
        val_patterns.tolist(), truncation=True, padding=True, max_length=512)
    
    # Create datasets
    train_dataset = IntentDataset(train_encodings, train_labels_fold)
    val_dataset = IntentDataset(val_encodings, val_labels_fold)
    
    # Create directories for this fold
    fold_results_dir = os.path.join(base_dir, f'results_fold_{fold+1}')
    fold_logs_dir = os.path.join(base_dir, f'logs_fold_{fold+1}')
    fold_model_dir = os.path.join(base_dir, f'intent_cf_model_fold_{fold+1}')
    
    # Create directories if they don't exist
    os.makedirs(fold_results_dir, exist_ok=True)
    os.makedirs(fold_logs_dir, exist_ok=True)
    os.makedirs(fold_model_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=fold_results_dir,          # output directory
        num_train_epochs=20,                 # total # of training epochs
        per_device_train_batch_size=9,       # batch size per device during training
        per_device_eval_batch_size=64,       # batch size for evaluation
        warmup_steps=500,                    # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                   # strength of weight decay
        logging_dir=fold_logs_dir,           # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,         # Load the best model (based on metric)
        metric_for_best_model="accuracy",    # Use 'accuracy' to choose the best
        greater_is_better=True,              # Higher accuracy = better
        save_total_limit=1,                  # Limit total checkpoints
    )
    
    # Initialize model for this fold
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(tag_dict))
    
    trainer = Trainer(
        model=model,
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=compute_metrics,     # define metrics function
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    fold_accuracy = eval_results['eval_accuracy']
    fold_accuracies.append(fold_accuracy)
    
    print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")
    
    # Save the best model across all folds
    if fold_accuracy > best_accuracy:
        best_accuracy = fold_accuracy
        best_model_checkpoint = trainer.state.best_model_checkpoint
        # Save the best model
        trainer.save_model(fold_model_dir)
        print(f"New best model saved with accuracy: {best_accuracy:.4f}")

# Print cross-validation results
print(f"\n{'='*50}")
print("K-Fold Cross-Validation Results")
print(f"{'='*50}")
print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
print(f"Std Accuracy: {np.std(fold_accuracies):.4f}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(f"Best Model Checkpoint: {best_model_checkpoint}")
