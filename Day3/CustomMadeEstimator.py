import numpy as np
import tensorflow as tf

# input function for training
def train_input_fun(features,labels,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    return dataset.shuffle(1000).repeat().batch(batch_size)

# input function for evaluation and prediction
def eval_input_fun(features,labels,batch_size):
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.batch(batch_size)

# Model which i want to build
def my_model_fn(features,labels,mode,params):
    # input layer
    net = tf.feature_column.input_layer(features,feature_columns=params['feature_columns'])
    # add hidden layers
    for units in params['hidden_units']:
        net = tf.layers.dense(net,units=units,activation=tf.nn.relu)
    # output layer
    logits = tf.layers.dense(net,units=params['n_classes'],activation=None)
    # compute predictions
    predicted_classes = tf.argmax(logits,1)
    if mode==tf.estimator.ModeKeys.PREDICT:
        predictions ={'class_ids':predicted_classes[:,tf.newaxis],'probabilities':tf.nn.softmax(logits),'logits':logits}
        return tf.estimator.EstimatorSpec(mode,predictions=predictions)
    # calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    # compute evaluation metrics
    accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')
    metrics={'accuracy':accuracy}
    tf.summary.scalar('accuracy',accuracy[1])
    if mode==tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
    # Train operations
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)

# Get input data
import pandas as pd
data = pd.read_csv('/Users/lpothabattula/Desktop/TensorFlow/Day1/iris.csv')
columns = ['sepal_length','sepal_width','petal_length','petal_width','species']
X,y = data[columns[:4]].values,data[columns[4:]].values

# change output labels from strings to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
labels = le.classes_

# split data into 70 and 30 percent
from sklearn.model_selection import train_test_split
X_train,X_test,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# get data into dictionary with column names as keys
from collections import defaultdict
train_x = defaultdict(list)
test_x = defaultdict(list)
for i in range(len(columns)-1):
    train_x[columns[i]] = X_train[:,i]
    test_x[columns[i]] = X_test[:,i]

# convert raw data to feature columns for model
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Create estimator and configure the model using params
clf = tf.estimator.Estimator(model_fn=my_model_fn,
                             params={'feature_columns':my_feature_columns,'hidden_units':[10,10],'n_classes':3})

#train model
clf.train(input_fn=lambda:train_input_fun(train_x,train_y,10),steps=1000)
# evaluate the model
eval_result = clf.evaluate(input_fn=lambda:eval_input_fun(test_x,test_y,10))
print('Test Accuracy {accuracy:0.3f}\n'.format(**eval_result))

# predict for new input
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
        'sepal_length': [5.1, 5.9, 6.9],
        'sepal_width': [3.3, 3.0, 3.1],
        'petal_length': [1.7, 4.2, 5.4],
        'petal_width': [0.5, 1.5, 2.1],
}

# Predict labels for unseen data
predictions = clf.predict(input_fn=lambda:eval_input_fun(predict_x,labels=None,batch_size=10))
for pred_dict,expc in zip(predictions,expected):
    cls_id = pred_dict['class_ids'][0]
    print('Predicted "{}" with probability:"{:.1f}, expected is "{}"'.format(labels[cls_id],pred_dict['probabilities'][cls_id] * 100,expc))
