
# import libaries that we use in our project
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# load data and use StandardScaler to standardize data

data = pd.read_csv("//home//knez//Documents//brest_cancer.csv",header=0,encoding="utf-8")
X = data.iloc[:,2:32].values
y = data.iloc[:,1]
y = np.where(y=='M',1,0)

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)
sc=StandardScaler()
#X_train_std = sc.fit_transform(X_train)
#X_test_std = sc.transform(X_test)
X_std=sc.fit_transform(X)
# covariance matrix, eigen values, eigen vectors
cov_mat = np.cov(X_std.T)
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)

# making pairs of eigen_vals and eigen_vecs

eigen_pairs = [(eigen_vals[i],eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k:k[0],reverse=True)
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))

# making pca data
X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.3,stratify=y,random_state=0)
X_train_pca = X_train.dot(w)
X_test_pca = X_test.dot(w)
colors = ["red","blue"]
markers = ["x","o"]



# plotting PCA data
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train==l,0],X_train_pca[y_train==l,1],color=c,marker=m,label=l)

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("Brest cancer data (using PCA)")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# making lda data
# calculating mean value of features

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(0,2):
    mean_vecs.append(np.mean(X_std[y==label],axis=0))


# calculating within class variance
d = X_std.shape[1]
S_W = np.zeros((d,d))
for label,mv in zip(range(0,2),mean_vecs):
    class_scatter = np.cov(X_std[y==label].T)
    S_W += class_scatter

# calculating between class variance
mean_overall = np.mean(X_std,axis=0)
S_B = np.zeros((d,d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_std[y==i,:].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    mean_overall = mean_overall.reshape(d,1)
    S_B+= n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
# calculating eigen vals and vecs for LDA
eigen_vals_lda,eigen_vecs_lda = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs_lda = [(np.abs(eigen_vals_lda[i]),eigen_vecs_lda[:,i]) for i in range(len(eigen_vals))]
eigen_pairs_lda = sorted(eigen_pairs_lda,key=lambda k:k[0],reverse=True)


# creating matrix of transformation
w_lda = np.hstack((eigen_pairs_lda[0][1][:,np.newaxis].real,eigen_pairs_lda[1][1][:,np.newaxis].real))


# ploting LDA data
X_train_lda = X_train.dot(w_lda)
X_test_lda = X_test.dot(w_lda)
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_lda[y_train==l,0],X_train_lda[y_train==l,1],c=c,marker=m,label=l)

plt.xlabel("LDA1")
plt.ylabel("LDA2")
plt.legend(loc="best")
plt.title("Brest cancer data (using LDA)")
plt.tight_layout()
plt.show()


# creating adaline algorithm with stochastic gradient descent optimization function

class AdalineSGD():
    def __init__(self,eta=0.01,n_iter=50,random_state=1,shuffle=True):
        self.eta=eta
        self.n_iter = n_iter
        self.random_state=random_state
        self.shuffle = shuffle

    def fit(self,X,y):

        self._initialize_weights(X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
               X,y = self._shuffle(X,y)
            cost =[]
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(cost)
        return self


    def _initialize_weights(self,m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0,scale=0.01,size=1+m)
        self.w_initialized = True

    def _shuffle(self,X,y):
        r = self.rgen.permutation(len(y))
        return X[r],y[r]

    def _update_weights(self,xi,target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:]+=self.eta*xi.dot(error)
        self.w_[0]+=self.eta*error
        cost = 0.5*error**2
        return cost
    
    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]
    
    def predict(self,X):
        return np.where(self.net_input(X)>=0.5,1,0)
    
    def activation(self,X):
        return X


# using adaline algorithm with PCA DATA
adaSGD = AdalineSGD(eta=0.001,n_iter=1000,random_state=1)
adaSGD.fit(X_train_pca,y_train)

# making function for ploting descision regions

def plot_descision_regions(X,y,classifier,resolution=0.02):
    colors=["red","blue","green","yellow","lightgreen"]
    markers = ["x","o","^","*","+"]
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max = X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max = X[:,1].min()-1,X[:,1].max()+1

    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],alpha=0.8,color=colors[idx],marker=markers[idx],label=cl,edgecolor="black")

plot_descision_regions(X_train_pca,y_train,classifier=adaSGD)
plt.title("AdalineSGD algorithm descision regions (PCA)")
plt.xlabel("First mark")
plt.ylabel("Second mark")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# print result of adalineSGD algorithm using PCA
from sklearn.metrics import accuracy_score
y_test_pred_adaline = adaSGD.predict(X_test_pca)
print("Result of adaline algorithm for test data (PCA): %.3f"  %(accuracy_score(y_test,y_test_pred_adaline)))

# use adaline algorithm for LDA data
adaSGD_LDA = AdalineSGD(eta=0.001,n_iter=1000,random_state=1)
adaSGD_LDA.fit(X_train_lda,y_train)

plot_descision_regions(X_train_lda,y_train,classifier=adaSGD_LDA)
plt.title("AdalineSGD algorithm descision regions (LDA)")
plt.xlabel("First mark")
plt.ylabel("Second mark")
plt.xlim([-0.9,0.7])
plt.ylim([-0.4,0.2])
plt.legend(loc="best")
plt.tight_layout()
plt.show()

#print result of adalineSGD algorithm using LDA
y_test_pred_lda = adaSGD_LDA.predict(X_test_lda)
print("Result of adaline algorithm for test data (LDA): %.3f" %(accuracy_score(y_test,y_test_pred_lda)))


# Logistic regression algorithm for PCA data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
lr= LogisticRegression(penalty='l2',solver='lbfgs',random_state=1,max_iter=1000)
lr.fit(X_train_pca,y_train)

param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
param_grid = [{"C":param_range}]
gs = GridSearchCV(estimator=lr,param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)
gs.fit(X_train_pca,y_train)
plot_descision_regions(X_train_pca,y_train,classifier=lr)
plt.title("Logistic regression algorithm for PCA data")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
y_pred_test_lr = lr.predict(X_test_pca)
print("Result of logistic regression algorithm for test data (PCA): %.3f" %(accuracy_score(y_test,y_pred_test_lr)))


# Logistic regression algorithm for LDA data

lr_lda = LogisticRegression(penalty='l2',solver='lbfgs',random_state=1,max_iter=1000)
lr_lda.fit(X_train_lda,y_train)
gs = GridSearchCV(estimator=lr_lda,param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)
gs.fit(X_train_pca,y_train)

plot_descision_regions(X_train_lda,y_train,classifier=lr_lda)
plt.title("Logistic regression algorithm for LDA data")
plt.legend(loc="best")
plt.xlim([-0.9,0.7])
plt.ylim([-0.4,0.2])
plt.tight_layout()
plt.show()


y_test_pred_lda_lr=lr_lda.predict(X_test_lda)
print("Result of logistic regression algorithm for test data (LDA): %.3f" %(accuracy_score(y_test,y_test_pred_lda_lr)))



# MAKING KNN ALGORITHM 
from sklearn.neighbors import KNeighborsClassifier

# THIS IS FOR PCA DATA
knn = KNeighborsClassifier()
knn.fit(X_train_pca,y_train)
param_neigh = [1,2,3,4,5]
param_met = ["minkowski","euclidean"]

param_grid = [{"n_neighbors":param_neigh,"metric":param_met}]
gs = GridSearchCV(estimator=knn,param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)
gs.fit(X_train_pca,y_train)
# PLOT DESCISION REGIONS FOR KNN (PCA)
plot_descision_regions(X_train_pca,y_train,classifier=knn)
plt.xlabel("First mark")
plt.ylabel("Second mark")
plt.title("KNN algorithm using PCA data")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
y_pred_test_knn_pca=knn.predict(X_test_pca)
print("Result of KNN algorithm using PCA on test data is: %.3f" %(accuracy_score(y_test,y_pred_test_knn_pca)))



# AND THIS IS KNN FOR LDA DATA
knn_lda = KNeighborsClassifier()
knn_lda.fit(X_train_lda,y_train)
gs = GridSearchCV(estimator=knn_lda,param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)
gs.fit(X_train_lda,y_train)
# PLOT DESCISION REGIONS FOR KNN (LDA)  
plot_descision_regions(X_train_lda,y_train,classifier=knn_lda)
plt.xlabel("First mark")
plt.ylabel("Second mark")
plt.title("KNN algorithm using LDA data")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
y_pred_test_knn_lda = knn_lda.predict(X_test_lda)
print("Result of KNN algorithm using LDA on test data is: %.3f" %(accuracy_score(y_test,y_pred_test_knn_lda)))

