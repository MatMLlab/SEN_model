
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



val_pre = pd.read_csv('val_prediction_64_epoch.csv', header=None)
val_real = pd.read_csv('val_ground_truth_data.csv', header=None, sep=",")

train_pre = pd.read_csv('training_prediction_64_epoch.csv', header=None)
train_real = pd.read_csv('training_ground_truth_data.csv', header=None, sep=",")


val_pre = np.array(val_pre).reshape(-1)
val_real = np.array(val_real).reshape(-1)
train_pre = np.array(train_pre).reshape(-1)
train_real = np.array(train_real).reshape(-1)

val_mae = np.mean(np.abs(val_pre - val_real))
train_mae = np.mean(np.abs(train_pre - train_real))


# Training error plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
plt.scatter(x = train_real,y = train_pre, label = 'Train', color='royalblue', alpha=.5)
#plt.scatter(x = train_pre,y = train_real,label = 'train', color='tomato', alpha=.5)
#ax.legend(loc = 'lower right', frameon = False,prop={'family' : 'Calibri', 'size' : 28})
plt.yticks([0, 10],fontproperties = 'Calibri',size = 16)
plt.xticks([0, 10],fontproperties = 'Calibri', size = 16)
#plt.xlim(0, 1)
#plt.ylim(0, 1)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.show()

# Val error plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
plt.scatter(x = val_real,y = val_pre, label = 'val', color='tomato', alpha=.5)
#plt.scatter(x = train_pre,y = train_real,label = 'train', color='tomato', alpha=.5)
#ax.legend(loc = 'lower right', frameon = False,prop={'family' : 'Calibri', 'size' : 28})
plt.yticks([0, 10],fontproperties = 'Calibri',size = 16)
plt.xticks([0, 10],fontproperties = 'Calibri', size = 16)
#plt.xlim(0, 1)
#plt.ylim(0, 1)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.show()



print("A")