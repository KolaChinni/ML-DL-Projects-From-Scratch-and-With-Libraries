import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sb

#----------callback to stop training -----------------

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('accuracy')>0.98:
            print('model reached 98% accuracy')
            self.model.stop_training=True

callbacks=myCallback()

#----------model building and training----------

model=Sequential(
    [
        Dense(units=30,activation='relu',input_shape=(30,)),
        Dense(units=8,activation='relu'),
        Dense(units=1,activation='sigmoid')
    ]
)

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#----------data loading and preprocessing----------

data=pd.read_csv('data.csv')
#print(data.info())
data=data.drop(['id','Unnamed: 32'],axis=1)

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

x=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']
x=(x-x.min())/(x.max()-x.min())

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


fits=model.fit(x_train,y_train,epochs=100,validation_split=0.2,callbacks=[callbacks])

#----------making predictions and evaluating----------

y_predict=model.predict(x_test)
y_predict=(y_predict>0.5)
loss,acc=model.evaluate(x_test,y_test)
print(f'loss = {loss}  and accuracy = {acc}')
classi_report=classification_report(y_test,y_predict)
confuse_matrix=confusion_matrix(y_test,y_predict)
print(f'classification report : \n {classi_report}')
print(f'confusion matrix : \n {confuse_matrix}')

#----------plotting accuracy and confusion matrix----------

plt.plot(fits.history['val_accuracy'],label='val_accuracy')
plt.plot(fits.history['accuracy'],label='train_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('train_accuracy vs validation_accuracy')
plt.legend()
plt.grid(True)
plt.show()


sb.heatmap(confuse_matrix,annot=True,fmt='d',cmap='Blues')
plt.title('confusion matrix')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()