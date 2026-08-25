import numpy as np
import tensorflow as tf

data=tf.keras.datasets.fashion_mnist
(trainimg,trainlabel),(testimg,testlabel)=data.load_data()
trainimg=trainimg/255.0
testimg=testimg/255.0import numpy as np
import tensorflow as tf

#-----Loading dataset-----

data=tf.keras.datasets.fashion_mnist
(trainimg,trainlabel),(testimg,testlabel)=data.load_data()
trainimg=trainimg/255.0
testimg=testimg/255.0
y_train = np.eye(10)[trainlabel]
y_test  = np.eye(10)[testlabel]
x_train=trainimg.reshape(trainimg.shape[0],-1)
x_test=testimg.reshape(testimg.shape[0],-1)

#----------activations----------

def relu(z):
    return np.maximum(0,z)
def relu_derivative(z):
    return np.where(z<=0,0,1)
def softmax(z):
    z=z-np.max(z,axis=1,keepdims=True)
    exp_z=np.exp(z)
    return exp_z/np.sum(exp_z,axis=1,keepdims=True)
def softmax_derivative(z):
    z=z.reshape(-1,1)
    return np.diagflat(z)-np.dot(z,z.T)

#----------mini-batch creation----------

def mini_batch(x,y,batch_size):
    m=x.shape[0]
    batches=[]
    for batch in range(0,m,batch_size):
        x_batch=x[batch:batch+batch_size]
        y_batch=y[batch:batch_size+batch]
        batches.append([x_batch,y_batch])
    return batches

#----------dense layer----------

def Dense(w,b,a_prev,activation):
    z=np.dot(a_prev,w)+b
    if activation=='relu':
        a_out=relu(z)
    elif activation=='softmax':
        a_out=softmax(z)
    else:
        a_out=z
    return a_out

#----------sequential model----------

def Sequential(x_train,W1,b1,W2,b2):
    a1=Dense(W1,b1,x_train,activation='relu')
    a2=Dense(W2,b2,a1,activation='softmax')
    return a1,a2

#----------forward and backward propagation with adam optimizer----------

def Forward_Backward():
    W1=np.random.randn(784,128)*0.01
    b1=np.random.randn(1,128)*0.01
    W2=np.random.randn(128,10)*0.01
    b2=np.random.randn(1,10)*0.01

    v_dw1,v_dw2=np.zeros_like(W1),np.zeros_like(W2)
    v_db1,v_db2=np.zeros_like(b1),np.zeros_like(b2)
    s_dw1,s_dw2=np.zeros_like(W1),np.zeros_like(W2)
    s_db1,s_db2=np.zeros_like(b1),np.zeros_like(b2)
    alpha=0.001
    beta1,beta2=0.9,0.999
    eps=1e-8
    t=0
    iter=100
    for epoch in range(iter):
        num_batches=0
        batches_loss=0
        for batch in mini_batch(x_train,y_train,64):
            trainimg,trainlabel_oh=batch[0],batch[1]
            a1,a2=Sequential(trainimg,W1,b1,W2,b2)
            loss = -np.mean(np.sum(trainlabel_oh * np.log(a2 + 1e-8), axis=1))
            num_batches+=1
            batches_loss+=loss
            dldz2=a2-trainlabel_oh
            dw2=np.dot(a1.T,dldz2)
            db2=np.sum(dldz2,axis=0,keepdims=True)

            dldz1=np.dot(dldz2,W2.T)*relu_derivative(a1)
            dw1=np.dot(trainimg.T,dldz1)
            db1=np.sum(dldz1,axis=0,keepdims=True)

            #----------------Adam--------------------
            t+=1

            bc1=1-beta1**t
            v_dw1=v_dw1*beta1+(1-beta1)*dw1
            v_dw1_corr=v_dw1/bc1
            v_dw2=v_dw2*beta1+(1-beta1)*dw2
            v_dw2_corr=v_dw2/bc1
            v_db1=v_db1*beta1+(1-beta1)*db1
            v_db1_corr=v_db1/bc1
            v_db2=v_db2*beta1+(1-beta1)*db2
            v_db2_corr=v_db2/bc1


            bc2=1-beta2**t
            s_dw1=s_dw1*beta2+(1-beta2)*dw1**2
            s_dw1_corr=s_dw1/bc2
            s_dw2=s_dw2*beta2+(1-beta2)*dw2**2
            s_dw2_corr=s_dw2/bc2
            s_db1=s_db1*beta2+(1-beta2)*db1**2
            s_db1_corr=s_db1/bc2
            s_db2=s_db2*beta2+(1-beta2)*db2**2
            s_db2_corr=s_db2/bc2

            #------------------Parameters Update--------------------
            
            W1=W1-alpha*v_dw1_corr/(np.sqrt(s_dw1_corr)+eps)
            W2=W2-alpha*v_dw2_corr/(np.sqrt(s_dw2_corr)+eps)
            b1=b1-alpha*v_db1_corr/(np.sqrt(s_db1_corr)+eps)
            b2=b2-alpha*v_db2_corr/(np.sqrt(s_db2_corr)+eps)
        
        _,a2_test=Sequential(x_train,W1,b1,W2,b2)
        predictions=np.argmax(a2_test,axis=1)
        avg_loss=batches_loss/num_batches
        acc=np.mean(predictions==y_train.argmax(axis=1))
        print(f'Epoch [{epoch+1}/{iter}] | loss : {avg_loss:.4f} | accuracy : {acc*100:.2f}%')
        if acc>0.95:
            print(f'\nReached 95% accuracy at epoch {epoch}')
            break


    return W1,W2,b1,b2

W1,W2,b1,b2=Forward_Backward()

a1_test,a2_test=Sequential(x_test,W1,b1,W2,b2)
predictions=np.argmax(a2_test,axis=1)
accuracy=np.mean(predictions==testlabel)
    
print(f'Test Accuracy: {accuracy*100:.2f}%')

y_train = np.eye(10)[trainlabel]
y_test  = np.eye(10)[testlabel]
x_train=trainimg.reshape(trainimg.shape[0],-1)
x_test=testimg.reshape(testimg.shape[0],-1)