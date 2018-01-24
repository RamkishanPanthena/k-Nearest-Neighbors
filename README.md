
## K-nn Implementation

Trained and tested k-nn clasification that does the following:
    
i) Compute pairwise similarity using either dot/cosine/euclidean measures

ii) Find the k-nearest neighbors and predict the class of the test record

iii) Compute accuracy of the dataset

iv) Print the nearest neighbors given a particular index

Dataset used: MNIST (digit images)
https://en.wikipedia.org/wiki/MNIST_database

http://yann.lecun.com/exdb/mnist/

MNIST Parser used:
https://github.com/datapythonista/mnist


```python
import mnist
import numpy as np
import time
```

### Implementing k-nn


```python
# knn
class knn:
    def __init__(self, k, sim_metric):
        self.sim_matrix = None
        self.tmp_matrix = None
        self.k = k
        self.sim_metric = sim_metric
        self.indices = []
        self.values = []
        self.categories = []
        self.pred = []
        self.chunks = []
    
    def kneighbors(self, train_X, train_y, test_X):
        tic = time.process_time()
        chunks = [test_X[i:i+5000] for i in range(0, len(test_X), 5000)]
        self.chunks = chunks
        
        for i in range(len(chunks)):
            print ("Processing chunk", i)
            self.tmp_matrix = self.build_sim_matrix(train_X, chunks[i], self.sim_metric)
            
            for j in range(len(chunks[i])):
                test_record = self.tmp_matrix[:,j]
                kindices = np.argpartition(test_record, -self.k)[-self.k:]
                kvalues = test_record[kindices]
                kcategories = train_y[kindices]
                
                tmp = list(kcategories)
                
                self.indices.append(kindices)
                self.values.append(kvalues)
                self.categories.append(kcategories)
                self.pred.append(max(set(tmp), key = tmp.count))
        toc = time.process_time()
        print("Total time elapsed:", toc-tic, "seconds.")
    
    def find_nearest_neighbor_idx(self, idx, labels):
        print("Indices:")
        for i in self.indices[idx]:
            print(i)
        
        print("\nCategories:")
        for i in self.categories[idx]:
            print(i)

        print("\nPrediction:", self.pred[idx])
        
        print("\nActual label:", labels[idx])        
                
    def build_sim_matrix(self, train_X, test_X, metric):
        if metric == 'dot':
            dist = self.dot_product_sim_matrix(train_X, test_X)
        if metric == 'cosine':
            dist = self.cosine_sim_matrix(train_X, test_X)
        if metric == 'euclidean':
            dist = self.euclidean_dist_matrix(train_X, test_X)
        
        return dist
    
    def accuracy(self, pred, test_y):
        correct = 0
        for i in range(len(pred)):
            if pred[i] == test_y[i]:
                correct += 1
        
        return correct/len(pred)
    
    def dot_product_sim_matrix(self, train_X, test_X):
        dist = np.dot(train_X, test_X.T)
        return dist
    
    def cosine_sim_matrix(self, train_X, test_X):
        norm_train = np.linalg.norm(train_X, axis = 1, keepdims = True)
        norm_test = np.linalg.norm(test_X, axis = 1, keepdims = True)
        
        a = np.dot(train_X, test_X.T)
        b = np.dot(norm_train, norm_test.T)
        dist = a/b
        
        return dist
    
    def euclidean_dist_matrix(self, train_X, test_X):
        train_X = np.array(train_X)
        test_X = np.array(test_X)
        train_square = np.sum(np.square(train_X), axis = 1)
        test_square = np.sum(np.square(test_X), axis = 1)
        
        mul = np.dot(train_X, test_X.T)
        dist = np.sqrt(train_square[:, np.newaxis] + test_square - 2 * mul)
        
        return dist
```

### Processing data for MNIST


```python
images_train = mnist.train_images()
images_test = mnist.test_images()
```

### Converting to 2-d array


```python
train_X = images_train.reshape((images_train.shape[0], images_train.shape[1] * images_train.shape[2]))
test_X = images_test.reshape((images_test.shape[0], images_test.shape[1] * images_test.shape[2]))
train_y = mnist.train_labels()
test_y = mnist.test_labels()
```

### Normalizing the datasets


```python
train_X = train_X/255
test_X = test_X/255
```

### Running k-nn on MNIST Train set


```python
model1 = knn(5, 'cosine')
model1.kneighbors(train_X, train_y, train_X)
```

    Processing chunk 0
    Processing chunk 1
    Processing chunk 2
    Processing chunk 3
    Processing chunk 4
    Processing chunk 5
    Processing chunk 6
    Processing chunk 7
    Processing chunk 8
    Processing chunk 9
    Processing chunk 10
    Processing chunk 11
    Total time elapsed: 754.921875 seconds.
    

### Check accuracy on Train set


```python
print("Accuracy on Train set:", model1.accuracy(model1.pred, train_y))
```

    Accuracy on Train set: 0.9840833333333333
    

### Running k-nn on MNIST Test set


```python
model2 = knn(5, 'cosine')
model2.kneighbors(train_X, train_y, test_X)
```

    Processing chunk 0
    Processing chunk 1
    Total time elapsed: 123.53125 seconds.
    

### Checking accuracy on Test set


```python
print("Accuracy on Test set:", model2.accuracy(model2.pred, test_y))
```

    Accuracy on Test set: 0.9728
    

### Checking nearset neighbors given a particular index


```python
model2.find_nearest_neighbor_idx(10, test_y)
```

    Indices:
    34656
    9035
    48438
    13216
    49939
    
    Categories:
    0
    0
    0
    0
    0
    
    Prediction: 0
    
    Actual label: 0
    
