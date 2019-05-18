import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os



def input_data():
    input_file = 'Iris.csv'
    IRIS_fname = "data/"+input_file
    iris = pd.read_csv(IRIS_fname)

    #Replace categorical labels with numerical values
    iris.Species = iris.Species.replace(to_replace=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], value=[0, 1, 2])
    X = iris.drop(labels=['Id', 'Species'], axis=1).values
    X.astype(np.float32)
    Y = iris.Species.values.astype(np.int32)
    #Create 80% training and 20% test
    train_index = np.random.choice(len(X), int(round(len(X) * 0.8)), replace=False)
    test_index = np.array(list(set(range(len(X))) - set(train_index)))
    Y_train = Y[train_index]
    Y_test = Y[test_index]
    X_train = normalize(X[train_index]) # Normalize training sets
    X_test = normalize(X[test_index]) # Normalize test sets
    return X_train, Y_train, X_test, Y_test

# min-max mormalization
def normalize(X):
    col_max = np.max(X, axis=0)
    col_min = np.min(X, axis=0)
    normX = np.divide(X - col_min, col_max - col_min)
    return normX

# former inference is now used for combining inputs
# Get the linear output of the network combining inputs
def combine_inputs(X):
    X1 = tf.nn.tanh(tf.matmul(X, W1) + b1)
    X2 = tf.nn.tanh(tf.matmul(X1, W2) + b2)
    Y_net_linear = tf.matmul(X2, W3) + b3
    return Y_net_linear

def inference(X):
    return tf.nn.softmax(combine_inputs(X))

def loss(X, Y):
    Yhat = combine_inputs(X)
    SoftCE = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Yhat, labels=Y)
    return tf.reduce_mean(SoftCE)

def train(total_loss):
    learning_rate = 0.1
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    goal = opt.minimize(total_loss)
    return goal

def evaluate(sess, Xtest, Ytest):
    Yhat = inference(X)
    #Return the index with the largest value across axis
    Ypredict = tf.argmax(Yhat, axis=1, output_type=tf.int32) #in [0,1,2]
    #Cast a boolean tensor to float32
    correct = tf.cast(tf.equal(Ypredict, Y), tf.float32)
    accuracy_graph = tf.reduce_mean(correct)
    accuracy = sess.run(accuracy_graph, feed_dict={X: Xtest, Y: Ytest})
    return accuracy

# Shuffle the training data
def reshuffle(X, Y):
    data_index = 0
    NData = len(X)
    perm_indices = np.arange(NData)
    np.random.shuffle(perm_indices)
    X = X[perm_indices]
    Y = Y[perm_indices]
    return X, Y

# Read next training batch
def read_next_batch(X, Y, batch_size, train_index=0):
    n_train_examples = len(X)
    if train_index + batch_size < n_train_examples:
        X_train_batch = X[train_index:train_index + batch_size]
        Y_train_batch = Y[train_index:train_index + batch_size]
        train_index = train_index + batch_size
        return X_train_batch, Y_train_batch, train_index
    else:
        return None, None, None

n_dim = 4 #Feature Vector dimension
n_classes = 3 #Number of classes
n_hidden = 32 #Number of Hidden nodes
X = tf.placeholder(dtype=tf.float32, shape=[None, n_dim])
Y = tf.placeholder(dtype=tf.int32, shape=(None, ))
# Weights form a matrix, of a feature column per output class.
W1 = tf.Variable(tf.random_normal(shape=[n_dim, n_hidden]), dtype=tf.float32)
b1 = tf.Variable(tf.random_normal(shape=(n_hidden,)), dtype=tf.float32)
W2 = tf.Variable(tf.random_normal(shape=[n_hidden, n_hidden]), dtype=tf.float32)
b2 = tf.Variable(tf.random_normal(shape=(n_hidden,)), dtype=tf.float32)
W3 = tf.Variable(tf.random_normal(shape=[n_hidden, n_classes]), dtype=tf.float32)
b3 = tf.Variable(tf.random_normal(shape=(n_classes,)), dtype=tf.float32)


# Session execution
batch_size = 30 # Training batch size
Xtrain, Ytrain, Xtest, Ytest = input_data() # Get the data samples
init = tf.global_variables_initializer()
saver = tf.train.Saver() 
with tf.Session() as sess:
    sess.run(init) # variables initialization
    total_loss = loss(X, Y)
    train_op = train(total_loss)

    # actual training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        Xtrain, Ytrain = reshuffle(Xtrain, Ytrain)
        train_index = 0
        train_loss = 0
        loss_trace_list = []
        Xtrain_batch, Ytrain_batch, train_index = read_next_batch(Xtrain, Ytrain, batch_size,train_index)
        while Xtrain_batch is not None:
            temp_loss, _ = sess.run([total_loss, train_op], feed_dict={X:Xtrain_batch, Y:Ytrain_batch})
            loss_trace_list.append(temp_loss)
            train_loss += temp_loss
            Xtrain_batch, Ytrain_batch, train_index = read_next_batch(Xtrain, Ytrain, batch_size,train_index)

        # see how the loss decreases
        if epoch % 10 == 0:
            train_acc = evaluate(sess, Xtrain, Ytrain)
            test_acc = evaluate(sess, Xtest, Ytest)
            print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1,train_loss, train_acc, test_acc))
            # save best epoch ckpt
            save_path = saver.save(sess,"saved_models/model.ckpt")
