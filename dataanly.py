import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
DATA_NAME = "offshore"
# KDE
def get_kde_pair(data1, data2, bandwidth=5, kernel='gaussian'):
    eps = 1e-6

    data_min = min(data1.min(), data2.min())
    data_max = max(data1.max(), data2.max())
    step = (data_max - data_min) / 1000
    # get kde
    kde1 = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(data1)
    kde2 = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(data2)
    # get boundary
    while True:
        p1, p2 = np.exp(kde1.score(np.array([data_min]).reshape(-1, 1))), np.exp(
            kde2.score(np.array([data_min]).reshape(-1, 1)))
        p = max(p1, p2)
        if p < eps:
            break
        else:
            data_min = data_min - step
    while True:
        p1, p2 = np.exp(kde1.score(np.array([data_max]).reshape(-1, 1))), np.exp(
            kde2.score(np.array([data_max]).reshape(-1, 1)))
        p = max(p1, p2)
        if p < eps:
            break
        else:
            data_max = data_max + step
    x = np.linspace(data_min, data_max, 1000)
    # get curves
    kde1_curve = np.exp(kde1.score_samples(x.reshape(-1, 1)))
    kde2_curve = np.exp(kde2.score_samples(x.reshape(-1, 1)))
    return x, kde1_curve, kde2_curve

NUM_SAMPLES_PLOT = 1000 # number of samples used to plot
# 读取数据并加载到torch中
x_data = torch.tensor(np.load('../data/train_data1.npy')).to(dtype=torch.float32)
y_data = torch.tensor(np.squeeze(np.load('../data/val_data1.npy')[:, :, 0:1], axis=2)).to(dtype=torch.float32)
# 使用split_ratio进行数据集划分
split_ratio = 0.8  # 假设使用80%作为训练集
split_index = int(len(x_data) * split_ratio)

X_train, X_test = x_data[0:split_index], x_data[split_index:]
y_train, y_test = y_data[0:split_index], y_data[split_index:]

train_data = np.stack([X_train[:, :, 0]], axis=0)
test_data = np.stack([X_test[:, :, 0]], axis=0)
train_data = torch.from_numpy(train_data).float()
test_data = torch.from_numpy(test_data).float()

train_data = train_data.transpose(1, 2).contiguous().view(-1, X_train.size(-2))
test_data = test_data.transpose(1, 2).contiguous().view(-1, X_test.size(-2))

train_random_indices = np.random.choice(train_data.size(0), NUM_SAMPLES_PLOT, replace=False)
test_random_indices = np.random.choice(test_data.size(0), NUM_SAMPLES_PLOT, replace=False)
train_data_sample = train_data[train_random_indices].numpy()
test_data_sample = test_data[test_random_indices].numpy()

# concatenate all data
all_data = np.concatenate([train_data_sample, test_data_sample], axis=0)
data_tsne = TSNE(n_components=2, random_state=0).fit_transform(all_data)
# get each data
train_tsne = data_tsne[:NUM_SAMPLES_PLOT]
test_tsne = data_tsne[NUM_SAMPLES_PLOT:2*NUM_SAMPLES_PLOT]
print(train_tsne.shape)
up_x, up_kde1, up_kde2 = get_kde_pair(train_tsne[:, [0]], test_tsne[:, [0]])
right_x, right_kde1, right_kde2 = get_kde_pair(train_tsne[:, [1]], test_tsne[:, [1]])

x_min, x_max = up_x.min(), up_x.max()
y_min, y_max = right_x.min(), right_x.max()
# plt.gca().set_facecolor('#eeeefe')
plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c='#1744a8', label='Training', edgecolors='#11317c', alpha=0.2)
plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c='#b90104', label='Testing', edgecolors='#7e0004', alpha=0.2)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.legend(title="Period", fontsize=15, markerscale=1.5, title_fontsize=15, alignment="right")
# save
plt.savefig('kde_tsne_{0}.pdf'.format(DATA_NAME), dpi=300, bbox_inches='tight')
plt.show()

# plt upper kde
## set size
plt.figure(figsize=(7, 1.5))
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.plot(up_x, up_kde1, label='train', c='#1744a8', alpha=0.8)
plt.plot(up_x, up_kde2, label='test', c='#b90104', alpha=0.8)
# plt.xticks([])
plt.yticks([])
plt.xlim(x_min, x_max)
plt.ylim(0, max(max(up_kde1), max(up_kde2)) * 1.1)
plt.fill_between(x=up_x, y1=0, y2=up_kde1, facecolor='#1744a8', alpha=0.3)
plt.fill_between(x=up_x, y1=0, y2=up_kde2, facecolor='#b90104', alpha=0.3)
# save
plt.savefig('kde_tsne_{0}_up.pdf'.format(DATA_NAME), dpi=300, bbox_inches='tight')
plt.show()

# plt right kde
## set size
plt.figure(figsize=(7, 1.5))
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.plot(right_x, right_kde1, label='train', c='#1744a8', alpha=0.8)
plt.plot(right_x, right_kde2, label='test', c='#b90104', alpha=0.8)
# plt.xticks([])
plt.yticks([])
plt.xlim(y_min, y_max)
plt.ylim(0, max(max(right_kde1), max(right_kde2)) * 1.1)
plt.fill_between(x=right_x, y1=0, y2=right_kde1, facecolor='#1744a8', alpha=0.3)
plt.fill_between(x=right_x, y1=0, y2=right_kde2, facecolor='#b90104', alpha=0.3)
# save
plt.savefig('kde_tsne_{0}_right.pdf'.format(DATA_NAME), dpi=300, bbox_inches='tight')
plt.show()