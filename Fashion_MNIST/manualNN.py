# import numpy as np
# import tensorflow as tf

# #-----Loading dataset-----

# data=tf.keras.datasets.fashion_mnist
# (trainimg,trainlabel),(testimg,testlabel)=data.load_data()
# trainimg=trainimg/255.0
# testimg=testimg/255.0
# y_train = np.eye(10)[trainlabel]
# y_test  = np.eye(10)[testlabel]
# x_train=trainimg.reshape(trainimg.shape[0],-1)
# x_test=testimg.reshape(testimg.shape[0],-1)

# #----------activations----------

# def relu(z):
#     return np.maximum(0,z)
# def relu_derivative(z):
#     return np.where(z<=0,0,1)
# def softmax(z):
#     z=z-np.max(z,axis=1,keepdims=True)
#     exp_z=np.exp(z)
#     return exp_z/np.sum(exp_z,axis=1,keepdims=True)
# def softmax_derivative(z):
#     z=z.reshape(-1,1)
#     return np.diagflat(z)-np.dot(z,z.T)

# #----------mini-batch creation----------

# def mini_batch(x,y,batch_size):
#     m=x.shape[0]
#     batches=[]
#     for batch in range(0,m,batch_size):
#         x_batch=x[batch:batch+batch_size]
#         y_batch=y[batch:batch_size+batch]
#         batches.append([x_batch,y_batch])
#     return batches

# #----------dense layer----------

# def Dense(w,b,a_prev,activation):
#     z=np.dot(a_prev,w)+b
#     if activation=='relu':
#         a_out=relu(z)
#     elif activation=='softmax':
#         a_out=softmax(z)
#     else:
#         a_out=z
#     return a_out

# #----------sequential model----------

# def Sequential(x_train,W1,b1,W2,b2):
#     a1=Dense(W1,b1,x_train,activation='relu')
#     a2=Dense(W2,b2,a1,activation='softmax')
#     return a1,a2

# #----------forward and backward propagation with adam optimizer----------

# def Forward_Backward():
#     W1=np.random.randn(784,128)*0.01
#     b1=np.random.randn(1,128)*0.01
#     W2=np.random.randn(128,10)*0.01
#     b2=np.random.randn(1,10)*0.01

#     v_dw1,v_dw2=np.zeros_like(W1),np.zeros_like(W2)
#     v_db1,v_db2=np.zeros_like(b1),np.zeros_like(b2)
#     s_dw1,s_dw2=np.zeros_like(W1),np.zeros_like(W2)
#     s_db1,s_db2=np.zeros_like(b1),np.zeros_like(b2)
#     alpha=0.001
#     beta1,beta2=0.9,0.999
#     eps=1e-8
#     t=0
#     iter=100
#     for epoch in range(iter):
#         num_batches=0
#         batches_loss=0
#         for batch in mini_batch(x_train,y_train,64):
#             trainimg,trainlabel_oh=batch[0],batch[1]
#             a1,a2=Sequential(trainimg,W1,b1,W2,b2)
#             loss = -np.mean(np.sum(trainlabel_oh * np.log(a2 + 1e-8), axis=1))
#             num_batches+=1
#             batches_loss+=loss
#             dldz2=a2-trainlabel_oh
#             dw2=np.dot(a1.T,dldz2)
#             db2=np.sum(dldz2,axis=0,keepdims=True)

#             dldz1=np.dot(dldz2,W2.T)*relu_derivative(a1)
#             dw1=np.dot(trainimg.T,dldz1)
#             db1=np.sum(dldz1,axis=0,keepdims=True)

#             #----------------Adam--------------------
#             t+=1

#             bc1=1-beta1**t
#             v_dw1=v_dw1*beta1+(1-beta1)*dw1
#             v_dw1_corr=v_dw1/bc1
#             v_dw2=v_dw2*beta1+(1-beta1)*dw2
#             v_dw2_corr=v_dw2/bc1
#             v_db1=v_db1*beta1+(1-beta1)*db1
#             v_db1_corr=v_db1/bc1
#             v_db2=v_db2*beta1+(1-beta1)*db2
#             v_db2_corr=v_db2/bc1


#             bc2=1-beta2**t
#             s_dw1=s_dw1*beta2+(1-beta2)*dw1**2
#             s_dw1_corr=s_dw1/bc2
#             s_dw2=s_dw2*beta2+(1-beta2)*dw2**2
#             s_dw2_corr=s_dw2/bc2
#             s_db1=s_db1*beta2+(1-beta2)*db1**2
#             s_db1_corr=s_db1/bc2
#             s_db2=s_db2*beta2+(1-beta2)*db2**2
#             s_db2_corr=s_db2/bc2

#             #------------------Parameters Update--------------------
            
#             W1=W1-alpha*v_dw1_corr/(np.sqrt(s_dw1_corr)+eps)
#             W2=W2-alpha*v_dw2_corr/(np.sqrt(s_dw2_corr)+eps)
#             b1=b1-alpha*v_db1_corr/(np.sqrt(s_db1_corr)+eps)
#             b2=b2-alpha*v_db2_corr/(np.sqrt(s_db2_corr)+eps)
        
#         _,a2_test=Sequential(x_train,W1,b1,W2,b2)
#         predictions=np.argmax(a2_test,axis=1)
#         avg_loss=batches_loss/num_batches
#         acc=np.mean(predictions==y_train.argmax(axis=1))
#         print(f'Epoch [{epoch+1}/{iter}] | loss : {avg_loss:.4f} | accuracy : {acc*100:.2f}%')
#         if acc>0.95:
#             print(f'\nReached 95% accuracy at epoch {epoch}')
#             break


#     return W1,W2,b1,b2

# W1,W2,b1,b2=Forward_Backward()

# a1_test,a2_test=Sequential(x_test,W1,b1,W2,b2)
# predictions=np.argmax(a2_test,axis=1)
# accuracy=np.mean(predictions==testlabel)
    
# print(f'Test Accuracy: {accuracy*100:.2f}%')

# y_train = np.eye(10)[trainlabel]
# y_test  = np.eye(10)[testlabel]
# x_train=trainimg.reshape(trainimg.shape[0],-1)
# x_test=testimg.reshape(testimg.shape[0],-1)
import numpy as np
import tensorflow as tf

# =========================================================
#                LOAD & PREPROCESS DATA
# =========================================================

# Load Fashion-MNIST dataset from TensorFlow
data = tf.keras.datasets.fashion_mnist
(trainimg, trainlabel), (testimg, testlabel) = data.load_data()

# Normalize pixel values to range [0, 1]
trainimg = trainimg / 255.0
testimg = testimg / 255.0

# One-hot encode labels (10 classes)
y_train = np.eye(10)[trainlabel]
y_test  = np.eye(10)[testlabel]

# Flatten images from (28, 28) → (784,)
x_train = trainimg.reshape(trainimg.shape[0], -1)
x_test  = testimg.reshape(testimg.shape[0], -1)

# =========================================================
#                ACTIVATION FUNCTIONS
# =========================================================

def relu(z):
    """ReLU activation"""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU"""
    return np.where(z <= 0, 0, 1)

def softmax(z):
    """Softmax activation for multiclass classification"""
    z = z - np.max(z, axis=1, keepdims=True)  # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# =========================================================
#                MINI-BATCH CREATION
# =========================================================

def mini_batch(x, y, batch_size):
    """
    Split dataset into mini-batches
    """
    m = x.shape[0]
    batches = []
    for batch in range(0, m, batch_size):
        x_batch = x[batch:batch + batch_size]
        y_batch = y[batch:batch + batch_size]
        batches.append([x_batch, y_batch])
    return batches

# =========================================================
#                DENSE LAYER
# =========================================================

def Dense(w, b, a_prev, activation):
    """
    Fully connected layer
    """
    z = np.dot(a_prev, w) + b
    if activation == 'relu':
        return relu(z)
    elif activation == 'softmax':
        return softmax(z)
    else:
        return z

# =========================================================
#                FORWARD PROPAGATION
# =========================================================

def Sequential(x, W1, b1, W2, b2):
    """
    Forward pass through the network
    """
    a1 = Dense(W1, b1, x, activation='relu')
    a2 = Dense(W2, b2, a1, activation='softmax')
    return a1, a2

# =========================================================
#                TRAINING (BACKPROP + ADAM)
# =========================================================

def Backward():
    """
    Train the neural network using:
    - Backpropagation
    - Adam optimizer (implemented from scratch)
    """

    # Weight initialization
    W1 = np.random.randn(784, 128) * 0.01
    b1 = np.random.randn(1, 128) * 0.01
    W2 = np.random.randn(128, 10) * 0.01
    b2 = np.random.randn(1, 10) * 0.01

    # Adam optimizer parameters
    v_dw1, v_dw2 = np.zeros_like(W1), np.zeros_like(W2)
    v_db1, v_db2 = np.zeros_like(b1), np.zeros_like(b2)
    s_dw1, s_dw2 = np.zeros_like(W1), np.zeros_like(W2)
    s_db1, s_db2 = np.zeros_like(b1), np.zeros_like(b2)

    alpha = 0.001
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0
    epochs = 100

    for epoch in range(epochs):
        num_batches = 0
        total_loss = 0

        for x_batch, y_batch in mini_batch(x_train, y_train, 64):
            # Forward pass
            a1, a2 = Sequential(x_batch, W1, b1, W2, b2)

            # Categorical cross-entropy loss
            loss = -np.mean(np.sum(y_batch * np.log(a2 + eps), axis=1))
            total_loss += loss
            num_batches += 1

            # Backpropagation
            dldz2 = a2 - y_batch
            dw2 = np.dot(a1.T, dldz2)
            db2 = np.sum(dldz2, axis=0, keepdims=True)

            dldz1 = np.dot(dldz2, W2.T) * relu_derivative(a1)
            dw1 = np.dot(x_batch.T, dldz1)
            db1 = np.sum(dldz1, axis=0, keepdims=True)

            # Adam update
            t += 1
            v_dw1 = beta1 * v_dw1 + (1 - beta1) * dw1
            v_dw2 = beta1 * v_dw2 + (1 - beta1) * dw2
            v_db1 = beta1 * v_db1 + (1 - beta1) * db1
            v_db2 = beta1 * v_db2 + (1 - beta1) * db2

            s_dw1 = beta2 * s_dw1 + (1 - beta2) * dw1**2
            s_dw2 = beta2 * s_dw2 + (1 - beta2) * dw2**2
            s_db1 = beta2 * s_db1 + (1 - beta2) * db1**2
            s_db2 = beta2 * s_db2 + (1 - beta2) * db2**2

            # Bias correction
            v_dw1_corr = v_dw1 / (1 - beta1**t)
            v_dw2_corr = v_dw2 / (1 - beta1**t)
            v_db1_corr = v_db1 / (1 - beta1**t)
            v_db2_corr = v_db2 / (1 - beta1**t)

            s_dw1_corr = s_dw1 / (1 - beta2**t)
            s_dw2_corr = s_dw2 / (1 - beta2**t)
            s_db1_corr = s_db1 / (1 - beta2**t)
            s_db2_corr = s_db2 / (1 - beta2**t)

            # Update parameters
            W1 -= alpha * v_dw1_corr / (np.sqrt(s_dw1_corr) + eps)
            W2 -= alpha * v_dw2_corr / (np.sqrt(s_dw2_corr) + eps)
            b1 -= alpha * v_db1_corr / (np.sqrt(s_db1_corr) + eps)
            b2 -= alpha * v_db2_corr / (np.sqrt(s_db2_corr) + eps)

        # Training accuracy
        _, a2_train = Sequential(x_train, W1, b1, W2, b2)
        preds = np.argmax(a2_train, axis=1)
        acc = np.mean(preds == trainlabel)
        avg_loss = total_loss / num_batches

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Accuracy: {acc*100:.2f}%")

        if acc > 0.95:
            print("\nReached 95% training accuracy. Stopping early.")
            break

    return W1, W2, b1, b2

# =========================================================
#                TRAIN & TEST
# =========================================================

W1, W2, b1, b2 = Backward()

# Test set evaluation
_, a2_test = Sequential(x_test, W1, b1, W2, b2)
predictions = np.argmax(a2_test, axis=1)
accuracy = np.mean(predictions == testlabel)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

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

def mini_batch(x,y,batch_size):
    m=x.shape[0]
    batches=[]
    for batch in range(0,m,batch_size):
        x_batch=x[batch:batch+batch_size]
        y_batch=y[batch:batch_size+batch]
        batches.append([x_batch,y_batch])
    return batches

def Dense(w,b,a_prev,activation):
    z=np.dot(a_prev,w)+b
    if activation=='relu':
        a_out=relu(z)
    elif activation=='softmax':
        a_out=softmax(z)
    else:
        a_out=z
    return a_out

def Sequential(x_train,W1,b1,W2,b2):
    a1=Dense(W1,b1,x_train,activation='relu')
    a2=Dense(W2,b2,a1,activation='softmax')
    return a1,a2



def Backward():
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
            print(f'\nReached 96% accuracy at epoch {epoch}')
            break


    return W1,W2,b1,b2

W1,W2,b1,b2=Backward()

a1_test,a2_test=Sequential(x_test,W1,b1,W2,b2)
predictions=np.argmax(a2_test,axis=1)
accuracy=np.mean(predictions==testlabel)
    
print(f'Test Accuracy: {accuracy*100:.2f}%')


