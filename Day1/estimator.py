# The Iris classification Problem
import pandas as pd
import tensorflow as tf
# input function
def train_input_fun(features,labels,batch_size):
    #Convert data to Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    # Shuffles, repeat and batch the data
    return dataset.shuffle(1000).repeat().batch(batch_size)
def eval_input_fun(features,labels,batch_size):
    features = dict(features)
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(features)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    # Shuffles, repeat and batch the data
    dataset = dataset.batch(batch_size)
    return dataset
#Get input data
data = pd.read_csv('/Users/lpothabattula/Desktop/TensorFlow/Day1/iris.csv')
X,y = data[['sepal_length','sepal_width','petal_length','petal_width']].values,data['species'].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
classes = le.classes_
from sklearn.model_selection import train_test_split
X_train,X_test,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=42)
train_x = {
    'SepalLength': X_train[:,0],
    'SepalWidth': X_train[:,1],
    'PetalLength': X_train[:,2],
    'PetalWidth': X_train[:,3],
    }
test_x = {
    'SepalLength': X_test[:,0],
    'SepalWidth': X_test[:,1],
    'PetalLength': X_test[:,2],
    'PetalWidth': X_test[:,3],
    }
#Feature columns describe how to use input
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#Build and train the model
clf = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,hidden_units=[10,10],n_classes=3)
#train model
clf.train(input_fn=lambda:train_input_fun(train_x,train_y,10),steps=1000)
# evaluate the model
eval_result = clf.evaluate(input_fn=lambda:eval_input_fun(test_x,test_y,10))
print('Test Accuracy {accuracy:0.3f}\n'.format(**eval_result))

# predict for new input
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
}
predictions = clf.predict(input_fn=lambda:eval_input_fun(predict_x,labels=None,batch_size=10))

for pred_dict,expc in zip(predictions,expected):
    cls_id = pred_dict['class_ids'][0]
    print('Predicted "{}" with probability:"{:.1f}, expected is "{}"'.format(classes[cls_id],pred_dict['probabilities'][cls_id] * 100,expc))



