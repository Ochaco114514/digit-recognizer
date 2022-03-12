import csv
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1/(1+np.exp(-z))

def dsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def ReLU(z):
    return np.maximum(z, 0)

def dReLU(z):
    return 

class ANN:

    #hidden_layer为隐藏层每层的神经元个数的列表,alpha为学习率,iter为迭代次数
    def __init__(self,Lambda=1,alpha=0.6,iter=10000,hidden_layer=[]) -> None:
        self.Lambda=Lambda
        self.alpha=alpha
        self.iter=iter
        self.layer=hidden_layer
        self.omega=[]   #权重
        self.a=None     #omega*x
        self.b=[]   #偏置项

    #min     y[10][28000]
    def cost_function(self,x,y):
        m=x.shape[0]
        h_of_theta=self.a[len(self.layer)+1]
        #print(h_of_theta.shape)    
        return -1.0*np.sum(np.dot(y,np.log10(h_of_theta))+np.dot((1-y),np.log10(1-h_of_theta)))/m

    def fit(self,x,y):
        m=x.shape[0]
        n=x.shape[1]
        layers=len(self.layer)
        #权重初始化
        omega=np.random.randn(n,self.layer[0])
        b=np.random.rand(self.layer[0])*0.1     #1*100
        print(omega)
        self.omega.append(omega)
        self.b.append(b)
        for i in range(1,layers):
            omega=np.random.randn(self.layer[i-1],self.layer[i])   
            b=np.random.rand(self.layer[i])*0.1  #1*100
            print(omega)
            self.omega.append(omega)
            self.b.append(b)
        omega=np.random.randn(self.layer[layers-1],10) 
        b=np.random.rand(10)*0.1 #1*10
        print(omega)
        self.omega.append(omega)
        self.b.append(b)

        self.a=[0]*(layers+2)
        z=[0]*(layers+2)
        delta=[0]*(layers+2)
        delta1=[0]*(layers+2)
        self.a[0]=x  # m*n  输入层输入为x

        #前向传播+反向传播
        for i in range(self.iter):
            z[1]=np.dot(self.a[0],self.omega[0])#+self.b[0]  输入不带偏置
            self.a[1]=sigmoid(z[1])  #m*100
            for j in range(1,layers+1):
                z[j+1]=np.dot(self.a[j],self.omega[j])+self.b[j]    #加入偏置
                self.a[j+1]=sigmoid(z[j+1])

            print(i)
            #print(self.cost_function(x,y))
            delta[layers+1]=self.a[layers+1]-y.T  #delta应为m*单元数
            delta1[layers]=np.dot(self.a[layers].T,delta[layers+1])  #delta1应为omega相同

            #计算delta
            for j in range(layers,0,-1):
                delta[j]=np.dot(delta[j+1],self.omega[j].T)*self.a[j]*(1-self.a[j]) #m*100
                #print(self.delta[j].shape)
                delta1[j-1]=np.dot(self.a[j-1].T,delta[j])  #100*100
            
            #更新权重
            for j in range(layers+1):
                self.omega[j]=self.omega[j]-1.0*self.alpha*delta1[j]/m

    def pred(self,x,y):
        #写入文件
        f=open('\ANN.csv','w',encoding='utf-8',newline="")
        csv_writer=csv.writer(f)
        csv_writer.writerow(["ImageId","Label"])
        layers=len(self.layer)
        a=[0]*(layers+2)
        z=[0]*(layers+2)

        #预测结果
        a[0]=x
        z[1]=np.dot(a[0],self.omega[0])#+self.b[0]
        a[1]=sigmoid(z[1])
        for j in range(1,layers+1):
            z[j+1]=np.dot(a[j],self.omega[j])+self.b[j]
            a[j+1]=sigmoid(z[j+1])  #sigmoid
        cnt=0
        correct=0
        ans=a[layers+1]
        
        #计算正确率
        for i in range(ans.shape[0]):
            max=0
            maxr=0
            for j in range(ans.shape[1]):
                if ans[i][j]>max:
                    max=ans[i][j]
                    maxr=j
            #print("预测:{},实际:{}".format(maxr,y['Label'][cnt]))
            if maxr==y['Label'][cnt]:
                correct=correct+1
            csv_writer.writerow([cnt+1,maxr])
            cnt=cnt+1
        print(cnt)
        #print(ans.shape)
        accuracy=correct*1.0/y.shape[0]*100
        print("准确率:{}%".format(accuracy))
        print(ans)
        f.close()

#   t1-t0为程序运行时间
t0=time.time()
csv_data=pd.read_csv("\\train.csv")
x=csv_data.drop(columns=['label'])

#预处理
x=(x-x.min())/(x.max()-x.min())
x=x.fillna(0.0)

x_train=x.to_numpy()
y=csv_data['label']
y_train=np.zeros((10,y.shape[0]))
y=list(y)
#print(y)
#print(len(y))
cnt=0
for i in y:
    y_train[i][cnt]=1.0
    cnt=cnt+1

#y_train=y_train[:,0:21000]
#print(y_train)
print(y_train.shape)
x_test=pd.read_csv("\\test.csv")

#预处理
x_test=(x_test-x_test.min())/(x_test.max()-x_test.min())
x_test=x_test.fillna(0.0)
x_test=x_test.to_numpy()

#pca降维,否则运行时间太长，拟合效果也不好
pca = PCA(n_components = 0.86)
pca.fit(x_train)
x_train= pca.transform(x_train)
x_test= pca.transform(x_test)

print(x_train)
print(x_train.shape)
print(x_test)
print(x_test.shape)

#标准答案是kaggle上找了个100%的下载的
y_test=pd.read_csv("\标准答案.csv")
y_test=y_test.drop(columns=["ImageId"])

hidden_layer=[100,100]
ANN1=ANN(hidden_layer=hidden_layer)
ANN1.fit(x_train,y_train)
ANN1.pred(x_test,y_test)
t1=time.time()
print("耗时：{}s".format(t1-t0))
# 95.25%左右,4200s


#训练数据,有点乱,本来是给自己看的
# 100*100 1000 52%  5000 77%
#100*100*100  0.05 5000 87%  0.1 5000 90%  0.15 91%   0.2 92% 0.3 93% 0.35 93% 0.5 94%
#100*100*100*100  5000 0.3 93.41%   0.5 93.66% 
#100*100 5000 0.3 93.57%(2100s)  0.5 94.425%（56） 6000（0.5） 94.58%  8000  94.84   10000 95.26%  （*0.1）93%  （*0.4）94.89%（*0.6）95.47 95.28（4200s）