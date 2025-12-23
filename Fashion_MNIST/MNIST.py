import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#----------callback----------

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=()):
        if(logs.get('accuracy')>0.95):
            print(r'\n reached 95% accuracy')
            self.model.stop_training = True
callbacks=myCallback()

#----------loading data and processing----------

data=tf.keras.datasets.fashion_mnist
(trainimg,trainlabel),(testimg,testlabel)=data.load_data()
trainimg=trainimg/255.0
testimg=testimg/255.0

#----------model----------

model=Sequential([
    Flatten(input_shape=(28,28)),
    Dense(units=128,activation=tf.nn.relu),
    Dense(units=10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy' , metrics=['accuracy'])
model.fit(trainimg,trainlabel,epochs=50,callbacks=[callbacks])
loss,accuracy=model.evaluate(testimg,testlabel)
cl=model.predict(testimg)
#print(cl[0])
#print(testlabel[0])

#-------metrics--------

print(f'classification report : \n {classification_report(testlabel,cl.argmax(axis=1))}')
print(f'confusin matrix : \n {confusion_matrix(testlabel,cl.argmax(axis=1))}')
print(f'accuracy : {accuracy_score(testlabel,cl.argmax(axis=1))}')
