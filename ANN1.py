import numpy as np
import tensorflow as tf
import pandas as pd

dataset = pd.read_csv("D:\Workspace\ML Course\ML A to Z\my_codes\Churn_Modelling.csv")
x = dataset.iloc[:,3:-1].values #features
y = dataset.iloc[:,-1].values #dependant variable vector

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))#fit_transform doesnt return the op as np array hence np.array is used

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#initializing the ANN

ann = tf.keras.models.Sequential()

#adding i/p layer and first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#o/p layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compiling the ANN

ann.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

#train ANN

ann.fit(x_train, y_train, batch_size= 32, epochs = 100)

#predictions

#print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

#predicting the test set

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#making confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
