import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils import lemmatize, bag_of_words, tokenize, stem

with open('intents.json','r') as f:
    intents = json.load(f)

all_words=[]
tags=[]
words_labeled=[] #word with corresponding tag

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        word = tokenize(pattern)  
        all_words.extend(word)    # extend instead of append, beacuse we don't want list os lists
        words_labeled.append((word,tag))     # word with meaning 


signs=['?','!','.',',']

all_words=[lemmatize(word) for word in all_words if word not in signs]
all_words = sorted(set(all_words)) # aplly as a set for get unique values

X_train=[]
y_train=[]

for (word,tag) in words_labeled:
    bag = bag_of_words(word,all_words)
    X_train.append(bag)
    
    label=tags.index(tag)
    y_train.append(label)
    
X_train = np.array(X_train)
y_train = np.array(y_train)

class Chatbotdata(Dataset):
    def __init__(self):
        self.n_samples = len(X_train) #number of samples is equal to amount of X_train elements, so amount of sentences
        self.x_data = X_train
        self.y_data = y_train
        
    #dataset(index)
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

    #Dataset
batch_size = 8
dataset = Chatbotdata()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle=True, num_workers=0)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        
        #activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        # input layer 
        output = self.layer1(x)
        output = self.relu(output)
        
        # hidden layer 
        output = self.layer2(output)
        output = self.relu(output)
        
        # output layer 
        output = self.layer3(output)
        # no softmax, because we apply cross-entropy loss later
        return output
    
#Hyperparameters:
batch_size = 8
hidden_size = 8
output_size = len(tags) # number of labels
input_size = len(X_train[0]) # all of the bog have the same size, we can just take first
learning_rate = 0.005
num_epochs=2000


print(f'input_size: {input_size}, output_size: {output_size}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.int64)
        
        # forward learning
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # backward learning and optimizer step
        optimizer.zero_grad() #apply this, because we don't want to sum gradient after epoch
        
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'epoch={epoch+1}/{num_epochs}, loss={loss.item():.2f}')
print(f'final loss: {loss.item():.2f}') 

data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}

FILE ='data.pth'
torch.save(data, FILE)
print(f'training complete, file saved to file {FILE}')