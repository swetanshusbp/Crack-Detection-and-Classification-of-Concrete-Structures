#!/usr/bin/env python
# coding: utf-8

# In[10]:


#importng necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam,RMSprop, Adagrad
from keras.layers import BatchNormalization
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[11]:


#LOAD DATASET
labels = ['Negative', 'Positive']
img_size = 120
def read_images(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

Dataset = read_images("C:\\Users\\Swetanshu's\\Desktop\\Crack Detection and Classification using CNN Model DL\\Dataset")


# In[12]:


import cv2
import numpy as np

image_path = "C:\\Users\\Swetanshu's\\Desktop\\Crack Detection and Classification using CNN Model DL\\Dataset\\Positive"

def find_crack_length_depth(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)

    min_angle = -20
    max_angle = 20
    crack_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if angle > min_angle and angle < max_angle:
            crack_lines.append(line)

    longest_line = max(crack_lines, key=lambda line: np.linalg.norm(line[0][:2] - line[0][2:]))
    shortest_lines = []
    for line in crack_lines:
        if np.cross(line[0][:2] - longest_line[0][:2], longest_line[0][:2] - line[0][2:]) == 0:
            shortest_lines.append(line)

    depth = min(np.linalg.norm(line[0][:2] - line[0][2:]) for line in shortest_lines)
    length = np.linalg.norm(longest_line[0][:2] - longest_line[0][2:])

    return length, depth



# In[14]:


Im = []
for i in Dataset:
    if(i[1] == 0):
        Im.append("Negative")
    elif(i[1] == 1):
        Im.append("Positive")

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
sns.set_style('darkgrid')
axl = sns.countplot(x=Im)
axl.set_title("Number of Images")


# In[15]:


#NORMALISATION OF IMAGE DATA
x = []
y = []

for feature, label in Dataset:
    x.append(feature)
    y.append(label)

x = np.array(x).reshape(-1, img_size, img_size, 1)
x = x / 255
y = np.array(y)


# In[16]:


plt.subplot(1, 2, 1)
plt.imshow(x[1000].reshape(img_size, img_size), cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x[30000].reshape(img_size, img_size), cmap='gray')
plt.axis('off')


# In[17]:


#CNN MODEL
model = Sequential()
model.add(Conv2D(64,3,padding="same", activation="relu", input_shape = x.shape[1:]))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(128, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(2, activation="softmax"))

model.summary()


# In[18]:


#MODEL TRAINING
opt = Adam(lr=1e-5)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) 

history = model.fit(x, y, epochs = 15, batch_size = 128, validation_split = 0.25, verbose=1)


# In[19]:


print(history.history.keys())


# In[20]:


#GRAPHS
plt.figure(figsize=(12, 12))
plt.style.use('ggplot')
plt.subplot(2,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy of the Model')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train accuracy', 'validation accuracy'], loc='lower right', prop={'size': 12})

plt.subplot(2,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss of the Model')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train loss', 'validation loss'], loc='best', prop={'size': 12})


# In[21]:


#CLASSIFICATION REPORT
from sklearn.metrics import classification_report, confusion_matrix

# Make predictions on the data using the model
y_pred = model.predict(x)

# Convert predicted probabilities to class labels
y_pred = np.argmax(y_pred, axis=1)

# Generate classification report and confusion matrix
class_names = ['No Crack', 'Crack']
report = classification_report(y, y_pred, target_names=class_names)
matrix = confusion_matrix(y, y_pred)

print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(matrix)    


# In[ ]:




