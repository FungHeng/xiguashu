import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mse(*data):
    y, y_pre=data
    return np.sum((y-y_pre) ** 2)

def mae(*data):
    y, y_pre=data
    return np.sum(abs(y-y_pre))


def huber(*data, delta):
    y, y_pre=data
    loss = np.where(np.abs(y - y_pre) <= delta, ((y - y_pre) ** 2),
                    2 * delta * np.abs(y - y_pre) - (delta ** 2))
    return np.sum(loss)

def logcosh(*data):
    y, y_pre=data
    loss = np.log(np.cosh(y_pre - y))
    return np.sum(loss)

def quan(*data, gamma):
    y, y_pre=data
    loss = np.where(y >= y_pre, gamma*(np.abs(y-y_pre)), (1-gamma)*(np.abs(y-y_pre)))
    return np.sum(loss)

fig, ax = plt.subplots(1,1, figsize = (10,6.5))

target = np.repeat(0, 1000)
pred = np.arange(-10,10, 0.02)

# calculating loss function for all predictions.
loss_mse = [mse(target[i], pred[i]) for i in range(len(pred))]
loss_mae = [mae(target[i], pred[i]) for i in range(len(pred))]
loss_huber1 = [huber(target[i], pred[i], delta=0.5) for i in range(len(pred))]
loss_huber2 = [huber(target[i], pred[i], delta=1) for i in range(len(pred))]
loss_logcosh = [logcosh(target[i], pred[i]) for i in range(len(pred))]
loss_quan1 = [quan(target[i], pred[i], gamma=0.25) for i in range(len(pred))]


losses = [loss_mse, loss_mae, loss_huber1, loss_huber2, loss_logcosh, loss_quan1]
names = ['MSE', 'MAE','Huber(0.5)', 'Huber(0.1)', 'Log-cosh', 'Quantile (0.25)']
cmap = ['#d53e4f','#fc8d59','#fee08b','#e6f598','#99d594','#3288bd']

for lo in range(len(losses)):
    ax.plot(pred, losses[lo], label = names[lo], color= cmap[lo])
ax.set_xlabel('Predictions')
ax.set_ylabel('Loss')
ax.set_title("Loss with Predicted values")
ax.legend()
ax.set_ylim(bottom=0, top=40)
plt.show()