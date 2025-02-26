import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()
        self.k1 = nn.Parameter(torch.tensor(1.0))  # Learnable parameter k1
        self.k2 = nn.Parameter(torch.tensor(0.0))  # Learnable parameter k2

    def forward(self, x):
        return self.k1 * x + self.k2
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.custom_activation = CustomActivation()  # Use custom activation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First layer with custom activation
        z1 = self.fc1(x)
        a1 = self.custom_activation(z1)

        # Second layer with custom activation
        z2 = self.fc2(a1)
        a2 = self.custom_activation(z2)

        # Final output layer with softmax activation
        z3 = self.fc3(a2)
        a3 = self.softmax(z3)
        return a3
      
df = pd.read_csv('/content/drive/MyDrive/data.csv')
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)
X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # Use long for classification labels
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

input_size = X_train.shape[1]
print(X_train.shape)
hidden_size = 16
output_size = 2  # Binary classification, but we use softmax for class probabilities

model = NeuralNetwork(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification with softmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, X_train, y_train, epochs):
    loss_values = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return loss_values

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)  # Get predictions

    accuracy = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted)
    recall = recall_score(y_test, predicted)
    f1 = f1_score(y_test, predicted)
    return accuracy, precision, recall, f1


epochs = 1000
loss_values = train(model, criterion, optimizer, X_train, y_train, epochs)

# Final evaluation
accuracy, precision, recall, f1 = evaluate(model, X_test, y_test)
print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

plt.plot(range(epochs), loss_values)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()

print("Final Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

