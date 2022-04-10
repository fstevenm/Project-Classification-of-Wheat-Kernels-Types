import numpy as np
import matplotlib.pyplot as plt

def bin_enc(lbl):
  mi = min(lbl)
  length = len(bin(max(lbl)-mi+1)[2:])
  enc=[]

  for i in lbl:
    b=bin(i-mi)[2:].zfill(length)
    enc.append([int(n) for n in b])

  return enc  

def bin_dec(enc, mi=0):
  lbl=[]

  for e in enc:
    rounded=[int(round(x)) for x in e]
    string= ''.join(str(x) for x in rounded)
    num=int(string,2) + mi
    lbl.append(num)
    
  return lbl


'''one-hot encoding'''

import numpy as np
def onehot_enc(lbl, min_val=0):
  mi=min(lbl)
  enc=np.full((len(lbl),max(lbl)-mi+1), min_val, np.int8)
  for i, x in enumerate(lbl):
    enc[i, x-mi]=1
  return enc

def onehot_dec(enc, mi=0):
  return [np.argmax(e)+mi for e in enc]

'''Fungsi aktivasi sigmoid dan turunannya'''

def sig(X):
 return [1/(1+np.exp(-x)) for x in X]

def sigd(X):
 output=[]
 for i, x in enumerate(X):
   s = sig([x])[0]
   output.append(s*(1-s))

 return output

'''fungsi modeling (training backpropagation)'''
def bp_fit(X, target, layer_conf, max_epoch, max_error=.1, learn_rate=.1,print_per_epoch=100):
  nin=[np.empty(i) for i in layer_conf]

  n = [np.empty(j+1) if i<len(layer_conf)-1
      else np.empty(j) for i, j in enumerate(layer_conf)]
      
  w = np.array([np.random.rand(layer_conf[i]+1, layer_conf[i+1])
                for i in range(len(layer_conf)-1)])
  
  dw = [np.empty((layer_conf[i]+1, layer_conf[i+1]))
        for i in range(len(layer_conf)-1)]
        
  d = [np.empty(s) for s in layer_conf[1:]]
  din = [np.empty(s) for s in layer_conf[1:-1]]
  epoch = 0
  mse = 1

  for i in range(0, len(n)-1):
    n[i][-1]=1
  while (max_epoch == -1 or epoch<max_epoch) and mse>max_error:
    epoch +=1
    mse = 0
    for r in range(len(X)):
      n[0][:-1]=X[r]

      for L in range(1, len(layer_conf)):
        nin[L] = np.dot(n[L-1], w[L-1])
        n[L][:len(nin[L])]=sig(nin[L])

      e = target[r] - n[-1]
      mse += sum(e ** 2)
      d[-1]=e*sigd(nin[-1])
      dw[-1]=learn_rate * d[-1]*n[-2].reshape((-1,1))

      for L in range(len(layer_conf)-1, 1, -1):
        din[L-2]=np.dot(d[L-1], np.transpose(w[L-1][:-1]))
        d[L-2]=din[L-2]*np.array(sigd(nin[L-1]))
        dw[L-2]=(learn_rate*d[L-2])*n[L-2].reshape((-1,1))

      w += dw
    mse /= len(X)

  if print_per_epoch > -1 and epoch % print_per_epoch == 0:
    print(f'Epoch {epoch}, MSE: {mse}')

  return w, epoch, mse

'''fungsi pengujian back propagation'''

def bp_predict(X,w):
  n=[np.empty(len(i)) for i in w]
  nin=[np.empty(len(i[0])) for i in w]
  predict = []
  n.append(np.empty(len(w[-1][0])))

  for x in X:
    n[0][:-1]=x

    for L in range(0, len(w)):
      nin[L] = np.dot(n[L], w[L])
      n[L+1][:len(nin[L])] = sig(nin[L])

    predict.append(n[-1].copy())
  
  return predict


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import minmax_scale


#import some data to play with
iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#class_names = iris.target_names
print(iris.target)
seeds_dataset = np.loadtxt('seeds_dataset.txt')

data = seeds_dataset[:, :7]
labels = seeds_dataset[:, 7].reshape((data.shape[0]))

label = []
for i in range (len(labels)) :
  label.append(int(labels[i]))

X = data
y = np.array(label)
print(y)
class_names = ['Kama','Rosa','Canadian']
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=.33)

print(len(X_test))

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
