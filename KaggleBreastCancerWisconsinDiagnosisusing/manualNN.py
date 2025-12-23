import numpy as np
import pandas as pd
from sklearn.model_selection  import train_test_split

#----------data loading and preprocessing----------

data=pd.read_csv(r'data.csv')
#print(data.info())
data=data.drop(['id','Unnamed: 32'],axis=1)

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

x=data.drop(['diagnosis'],axis=1)
y=data['diagnosis'].to_numpy()
x=(x-x.min())/(x.max()-x.min())  #---min max scaling

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)#split data

#----------reshaping y----------

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

#----------activation functions and their derivatives----------

def relu(z):
    return np.maximum(0,z)
def relu_derivative(a):
    return np.where(a<=0,0,1)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def sigmoid_derivative(a):
    return a*(1-a)

#---------Dense layer----------

def Dense(w,b,a_prev,activation):
    z=np.dot(a_prev,w)+b
    if activation=='relu':
        a_out=relu(z)
    elif activation=='sigmoid':
        a_out=sigmoid(z)
    else:
        a_out=z
    return a_out
    
#---------Sequential model----------

def Sequential(x_train,W1,W2,W3,b1,b2,b3):
    a1=Dense(W1,b1,x_train,activation='relu')
    a2=Dense(W2,b2,a1,activation='relu')
    a3=Dense(W3,b3,a2,activation='sigmoid')
    return a1,a2,a3

#------------forward and backward propagation with Adam optimizer----------

def Forward_Backward(x_train,y_train,W1,W2,W3,b1,b2,b3,iter):
    v_dw1,v_dw2,v_dw3=np.zeros_like(W1),np.zeros_like(W2),np.zeros_like(W3)
    v_db1,v_db2,v_db3=np.zeros_like(b1),np.zeros_like(b2),np.zeros_like(b3)
    s_dw1,s_dw2,s_dw3=np.zeros_like(W1),np.zeros_like(W2),np.zeros_like(W3)
    s_db1,s_db2,s_db3=np.zeros_like(b1),np.zeros_like(b2),np.zeros_like(b3)
    alpha=0.001
    beta1,beta2=0.9,0.999
    eps=1e-8
    for epoch in range(iter):
        a1,a2,a3=Sequential(x_train,W1,W2,W3,b1,b2,b3)
        loss=np.mean(-(y_train*np.log(a3+eps)+(1-y_train)*np.log(1-a3+eps))) #loss calculation
        if epoch%500==0:
            print(f'epoch {epoch} loss={loss}')
        dldz3=a3-y_train
        dw3=np.dot(a2.T,dldz3)
        db3=np.sum(dldz3,axis=0)

        dldz2=np.dot(dldz3,W3.T)*relu_derivative(a2)
        dw2=np.dot(a1.T,dldz2)
        db2=np.sum(dldz2,axis=0)

        dldz1=np.dot(dldz2,W2.T)*relu_derivative(a1)
        dw1=np.dot(x_train.T,dldz1)
        db1=np.sum(dldz1,axis=0)

        #-----------------Adam--------------------

        bc1=1-beta1**(epoch+1)
        v_dw1=v_dw1*beta1+(1-beta1)*dw1
        v_dw1_corr=v_dw1/bc1
        v_dw2=v_dw2*beta1+(1-beta1)*dw2
        v_dw2_corr=v_dw2/bc1
        v_dw3=v_dw3*beta1+(1-beta1)*dw3
        v_dw3_corr=v_dw3/bc1
        v_db1=v_db1*beta1+(1-beta1)*db1
        v_db1_corr=v_db1/bc1
        v_db2=v_db2*beta1+(1-beta1)*db2
        v_db2_corr=v_db2/bc1
        v_db3=v_db3*beta1+(1-beta1)*db3
        v_db3_corr=v_db3/bc1

        bc2=1-beta2**(1+epoch)
        s_dw1=s_dw1*beta2+(1-beta2)*dw1**2
        s_dw1_corr=s_dw1/bc2
        s_dw2=s_dw2*beta2+(1-beta2)*dw2**2
        s_dw2_corr=s_dw2/bc2
        s_dw3=s_dw3*beta2+(1-beta2)*dw3**2
        s_dw3_corr=s_dw3/bc2
        s_db1=s_db1*beta2+(1-beta2)*db1**2
        s_db1_corr=s_db1/bc2
        s_db2=s_db2*beta2+(1-beta2)*db2**2
        s_db2_corr=s_db2/bc2
        s_db3=s_db3*beta2+(1-beta2)*db3**2
        s_db3_corr=s_db3/bc2

        #------------------parameters update------------------

        W1=W1-alpha*v_dw1_corr/(np.sqrt(s_dw1_corr)+eps)
        W2=W2-alpha*v_dw2_corr/(np.sqrt(s_dw2_corr)+eps)
        W3=W3-alpha*v_dw3_corr/(np.sqrt(s_dw3_corr)+eps)
        b1=b1-alpha*v_db1_corr/(np.sqrt(s_db1_corr)+eps)
        b2=b2-alpha*v_db2_corr/(np.sqrt(s_db2_corr)+eps)
        b3=b3-alpha*v_db3_corr/(np.sqrt(s_db3_corr)+eps)

    return W1,W2,W3,b1,b2,b3

#-----------Network initialization----------

def Network_initailaize(x_train,y_train,iter):
    W1=np.random.rand(x_train.shape[1],30)
    b1=np.random.rand(30)
    W2=np.random.rand(30,8)
    b2=np.random.rand(8)
    W3=np.random.rand(8,1)
    b3=np.random.rand(1)
    W1,W2,W3,b1,b2,b3=Forward_Backward(x_train,y_train,W1,W2,W3,b1,b2,b3,iter)
    return W1,W2,W3,b1,b2,b3
    
W1,W2,W3,b1,b2,b3=Network_initailaize(x_train,y_train,iter=5000)

#-----------------------Prediction and Evaluation-----------------------

def predict(xpr):
    eps=1e-8
    
    z1=np.dot(xpr,W1)+b1
    a1=relu(z1)

    z2=np.dot(a1,W2)+b2
    a2=relu(z2)

    z3=np.dot(a2,W3)+b3
    a3=sigmoid(z3)

    return a3

#-----------------------Classification Report-----------------------

def classification_report(y_true,y_pred):
    classes=np.unique(y_true)
    report={}
    eps=1e-8
    for c in classes:
        tp=np.sum((y_true==c)&(y_pred==c))
        fp=np.sum((y_true!=c)&(y_pred==c))
        tn=np.sum((y_true!=c)&(y_pred!=c))
        fn=np.sum((y_true==c)&(y_pred!=c))

        precision=tp/(tp+fp+eps)
        recall=tp/(tp+fn+eps)
        f1_score=2*precision*recall/(precision+recall+eps)
        support=np.sum(y_true==c)
        report[c]={
            'precision':precision,
            'recall':recall,
            'f1_score':f1_score,
            'support':support
        }
    accuracy=np.sum(y_true==y_pred)/len(y_true)
    return report,accuracy

#-----------------------Confusion matrix-----------------------

def confusion_matrix(y_true,y_pred):
    y_true=np.array(y_true).flatten()
    y_pred=np.array(y_pred).flatten()
    num_classes=len(np.unique(np.concatenate((y_true,y_pred))))
    return np.bincount(num_classes*y_true+y_pred,minlength=num_classes**2).reshape(num_classes,num_classes)

#print(x_train.shape,x_test.shape)
pr=predict(x_test)
pr=pr>=0.5
report,accuracy=classification_report(y_test,pr)

print(confusion_matrix(y_test,pr))
print('-----------Report---------')
for cls,m in report.items():
    print(f"class : {cls} precision : {m['precision']} recall : {m['recall']} f1_score : {m['f1_score']} support : {m['support']}")
print('----')
print('accuracy :',accuracy)
