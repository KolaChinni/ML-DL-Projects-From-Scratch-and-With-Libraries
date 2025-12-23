import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler

#----------sigmoid function----------

def sigmoid(z):
    return 1/(1+np.exp(-z))

#----------gradient Calculation----------

def gradient(x,y,w,b):
    m,n=x.shape
    z=np.dot(x,w)+b
    f_wb=sigmoid(z)
    err=f_wb-y
    dj_dw=np.dot(x.T,err)
    dj_db=np.sum(err)
    dj_dw=dj_dw/m
    dj_db=dj_db/m
    return dj_dw,dj_db

#----------gradient descent----------
  
def gradient_descent(x,y,w,b,iter,alpha):
    for i in range(iter):
        dj_dw,dj_db=gradient(x,y,w,b)
        w=w-alpha*dj_dw
        b=b-alpha*dj_db
    return w,b

#----------predict function----------

def predict(x,w,b):
    z=np.dot(x,w)+b
    preds=sigmoid(z)
    return preds

#----------data loading and preprocessing----------

data=pd.read_csv('data.csv')
data=data.drop(['id','Unnamed: 32'],axis=1)
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
x=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']
#----------converting to nmpy arrays and scaling----------

x=x.to_numpy()
y=y.to_numpy()
scaler=StandardScaler()
x=scaler.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#----------initializing parameters and training----------

w=np.zeros(x.shape[1])
b=0.0
iter=1000
alpha=0.01
w,b=gradient_descent(x_train,y_train,w,b,iter,alpha)
#----------making predictions and evaluating----------
y_pred=predict(x_test,w,b)
y_pred=(y_pred>=0.5).astype(int)

#print(y_pred)

#----------evaluation metrics----------
classi_report=classification_report(y_test,y_pred)
confu_matrix=confusion_matrix(y_test,y_pred)
acc_score=accuracy_score(y_test,y_pred)
print('classification report \n',classi_report)
print('confusion matrix \n',confu_matrix)
print(f'accuracy : {acc_score}')