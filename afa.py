import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("/content/drive/MyDrive/data.csv")
print(data.head())

data.drop(columns=['Unnamed: 32'], inplace=True)
data.info()
data.isnull().sum()
data['diagnosis'].value_counts()

# Separate features (X) and labels (y)
X = data['radius_mean']
Y = data['diagnosis']

print(X)
data.replace(('B','M'), (0,1), inplace=True)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=3)
print(X.shape, X_train.shape, X_test.shape)
X = np.asarray(X)
Y = np.asarray(Y)

# list of models
models = [LogisticRegression(max_iter=1000), SVC(kernel='linear'), KNeighborsClassifier(), RandomForestClassifier(random_state=0)]

def compare_models_train_test():

  for model in models:

    # training the model
    model.fit(X_train, Y_train)

    # evaluating the model
    test_data_prediction = model.predict(X_test)

    accuracy = accuracy_score(Y_test, test_data_prediction)

    print('Accuracy score of the ', model, ' = ', accuracy)

compare_models_train_test()

def compare_models_cross_validation():

  for model in models:

    cv_score = cross_val_score(model, X, Y, cv=5)
    mean_accuracy = sum(cv_score)/len(cv_score)
    mean_accuracy = mean_accuracy*100
    mean_accuracy = round(mean_accuracy, 2)

    print('Cross Validation accuracies for the',model,'=', cv_score)
    print('Acccuracy score of the ',model,'=',mean_accuracy,'%')
    print('---------------------------------------------------------------')

compare_models_cross_validation()

cv_score_lr = cross_val_score(LogisticRegression(max_iter=1000), X, Y, cv=5)

print(cv_score_lr)

mean_accuracy_lr = sum(cv_score_lr)/len(cv_score_lr)

mean_accuracy_lr = mean_accuracy_lr*100

mean_accuracy_lr = round(mean_accuracy_lr, 2)

print(mean_accuracy_lr)

cv_score_svc = cross_val_score(SVC(kernel='linear'), X, Y, cv=5)

print(cv_score_svc)

mean_accuracy_svc = sum(cv_score_svc)/len(cv_score_svc)

mean_accuracy_svc = mean_accuracy_svc*100

mean_accuracy_svc = round(mean_accuracy_svc, 2)

print(mean_accuracy_svc)

                                #KNeighboursclassifier
cv_score_knc = cross_val_score(KNeighborsClassifier, X, Y, cv=5)

print(cv_score_knc)

mean_accuracy_knc = sum(cv_score_knc)/len(cv_score_knc)

mean_accuracy_knc = mean_accuracy_knc*100

mean_accuracy_knc = round(mean_accuracy_knc, 2)

print(mean_accuracy_knc)

cv_score_kFc = cross_val_score(RandomForestClassifier(random_state=0), X, Y, cv=5)

print(cv_score_kFc)

mean_accuracy_kFc = sum(cv_score_kFc)/len(cv_score_kFc)

mean_accuracy_kFc = mean_accuracy_kFc*100

mean_accuracy_kFc = round(mean_accuracy_kFc, 2)

print(mean_accuracy_kFc)

data.describe()

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

Y_pred_train = model.predict(X_train)
train_loss = mean_absolute_error(Y_train, Y_pred_train)
print("Train Loss:", train_loss)

y_pred_test = model.predict(X_test)
test_loss = mean_absolute_error(y_test, y_pred_test)
print("Test Loss:", test_loss)

from sklearn.metrics import accuracy_score
# accuracy on training data
X_train_prediction = models.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print(training_data_accuracy)

print('Accuracy on Training data : ', round(training_data_accuracy*100, 2), '%')

# accuracy on test data
X_test_prediction = models.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print(test_data_accuracy)

print('Accuracy on Test data : ', round(test_data_accuracy*100, 2), '%')

# precision for training data predictions
precision_train = precision_score(Y_train, X_train_prediction)
print('Training data Precision =', precision_train)

# precision for test data predictions
precision_test = precision_score(Y_test, X_test_prediction)
print('Test data Precision =', precision_test)

# recall for training data predictions
recall_train = recall_score(Y_train, X_train_prediction)
print('Training data Recall =', recall_train)

# F1 score of traing Data
F1_score= 2*(precision_train*recall_train)/(precision_train+recall_train)
print('Training data F1 Score =', F1_score)

# recall for test data predictions
recall_test = recall_score(Y_test, X_test_prediction)
print('Test data Recall =', recall_test)

# F1 score of test Data Predictations
F1_score= 2*(precision_train*recall_train)/(precision_train+recall_train)
print('Training data F1 Score =', F1_score)


#plot of loss function vs epochs
import matplotlib.pyplot as plt


# Sample data (replace with your actual training data)
#sample_loss_values = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
epochs = range(1, len(train_loss) + 1)

# Plot the loss curve
plt.plot(epochs, train_loss, marker='o', linestyle='-', color='blue')

# Customize the plot
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

# Show the plot
plt.show()