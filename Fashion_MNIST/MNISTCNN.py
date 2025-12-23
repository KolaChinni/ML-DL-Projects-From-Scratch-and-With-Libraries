import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#---------Callback----------
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=()):
        if logs.get('accuracy')>0.99:
            print('model reached 99% accuracy')
            self.model.stop_training=True
callbacks=mycallback()
#---------Loading data and processing----------
data=tf.keras.datasets.fashion_mnist
(trainimg,trainlabel),(testimg,testlabel)=data.load_data()
trainimg=trainimg.reshape(60000,28,28,1)
testimg=testimg.reshape(10000,28,28,1)
trainimg=trainimg/255.0
testimg=testimg/255.0

#---------Model----------

model=Sequential([
    Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(units=128,activation='relu'),
    Dense(units=10,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(trainimg,trainlabel,epochs=100,callbacks=[callbacks])
loss,accuracy=model.evaluate(testimg,testlabel)
pr=model.predict(testimg) 
#print(cl[0])
#print(testlabel[0])
#-------metrics--------
print(f'classification report : \n {classification_report(testlabel,pr.argmax(axis=1))}')
print(f'confusin matrix : \n {confusion_matrix(testlabel,pr.argmax(axis=1))}')
print(f'accuracy : {accuracy_score(testlabel,cl.argmax(axis=1))}')
