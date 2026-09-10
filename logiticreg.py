import numpy as np
import pandas as pd 
import matplotlib.pyplot as plot
#git test
#lets clear variables first here to avoid later confusion
# z=wx+b refer written 3b1b notes to revise 
def scale(x):
    x_mean=np.mean(x, axis=0)
    x_std=np.std(x, axis=0)
    x_std[x_std == 0] = 1e-8
    x_scaled=(x-x_mean)/x_std
    return x_mean,x_std,x_scaled
def sigmoid(z):
    sig=1.00/(1+np.exp(-z))
    return sig
def gradedes(xs,y,lr,epochs):
    m=len(xs)# we wish to avoid using b=0 then doing grade des on it to refer obsidian file to know more can be done traditianlly 
    b=np.c_[np.ones((xs.shape[0],1)),xs] #make a clumn matrix of 1s but in row formar
    w=np.zeros(b.shape[1]) # a row matrix
    for i in range(epochs):
        if i % 50 == 0:
            print(i)
        grad=np.dot(b.T,(sigmoid(np.dot(b,w))-y))/m
        w=w-lr*grad
    return w

def pred_prob(xs,w):
    b=np.c_[np.ones((xs.shape[0],1)),xs]
    return  sigmoid(np.dot(b,w))
def pred(xs,w):# taking direct input now after all the stuff is done
    return (pred_prob(xs,w)>=0.5).astype(int)

df=pd.read_csv('logistic_regression/candy-data.csv')
print("columns:",df.columns.tolist())
feature_cols = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat',
                'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent']
x=df[feature_cols].values
target=df['winpercent'].values
y=(target>50).astype(int)

xm,xstd,xs=scale(x)
lr=0.001
epochs=5000
w=gradedes(xs,y,lr,epochs)

ypred=pred(xs,w)


choclate=x[:,9]

plot.figure(figsize=(8,5))
plot.scatter(choclate, target, color='blue', alpha=0.5, label='winpercent')
probs = pred_prob(xs, w) * 100
plot.scatter(choclate, probs, color='red', alpha=0.5, label='Predicted probability (x100)')
plot.xlabel('Chocolate (0 = No, 1 = Yes)')
plot.ylabel('Winpercent / Predicted %')
plot.title('Actual vs Predicted — Chocolate')
plot.grid(True)
plot.show()