import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['samples']:
    tag = intent['tags']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:

        w = tokenize(pattern)

        all_words.extend(w)

        xy.append((w, tag))



# stem and lower each word
ignore_words = ['?', '!', '.', ',','is','are','am','was','were','do','does','did','can','could','may','might','must','shall','should','will','would','have','has','had','a','an','the','of','in','on','at','to','for','from','by','with','and','or','but','if','then','else','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now']
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))


# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:

    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 100
batch_size = 9
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = torch.tensor(X)
        self.y_data = torch.tensor(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# 5-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=321)
fold_accuracies = []
best_accuracy = 0.0
best_model_state = None
best_fold = 0
best_epoch = 0

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
    print(f'\nFold {fold + 1}/5')
    
    # Split data for this fold
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]
    
    # Create datasets and loaders
    train_dataset = ChatDataset(X_train_fold, y_train_fold)
    val_dataset = ChatDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    
    val_loader = DataLoader(dataset=val_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate the model on validation set
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for (words, labels) in val_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)
                outputs = model(words)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        
        # Save best model if current accuracy is better
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
            best_fold = fold + 1
            best_epoch = epoch + 1
            print(f'New best model! Fold {best_fold}, Epoch {best_epoch}, Accuracy: {best_accuracy:.4f}')
        
        if (epoch + 1) % 20 == 0:  # Print every 20 epochs to reduce output
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    # Final evaluation for this fold
    model.eval()
    final_predictions = []
    final_true_labels = []
    
    with torch.no_grad():
        for (words, labels) in val_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            _, predicted = torch.max(outputs.data, 1)
            final_predictions.extend(predicted.cpu().numpy())
            final_true_labels.extend(labels.cpu().numpy())
    
    fold_accuracy = accuracy_score(final_true_labels, final_predictions)
    fold_accuracies.append(fold_accuracy)
    print(f'Fold {fold + 1} Final Accuracy: {fold_accuracy:.4f}')

# Print cross-validation results
print(f'\nCross-Validation Results:')
print(f'Fold Accuracies: {[f"{acc:.4f}" for acc in fold_accuracies]}')
print(f'Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}')
print(f'Best Model: Fold {best_fold}, Epoch {best_epoch}, Accuracy: {best_accuracy:.4f}')

data = {
"model_state": best_model_state,
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Best model saved to {FILE}')
