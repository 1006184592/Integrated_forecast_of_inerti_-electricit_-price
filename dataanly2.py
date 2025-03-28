import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
DATA_NAME = "offshore"
data = pd.read_csv('../data/Offshore Wind Farm Dataset1(WT5).csv')
df = data.drop('Sequence No.', axis=1)
x_data = torch.tensor(np.load('../data/train_data1.npy')).to(dtype=torch.float32)
y_data = torch.tensor(np.squeeze(np.load('../data/val_data1.npy')[:, :, 0:1], axis=2)).to(dtype=torch.float32)
# 使用split_ratio进行数据集划分
split_ratio = 0.8  # 假设使用80%作为训练集
split_index = int(len(x_data) * split_ratio)

X_train, X_test = x_data[0:split_index], x_data[split_index:]
y_train, y_test = y_data[0:split_index], y_data[split_index:]

def plot_raw_data(data: np.array, selected_node_id: int, begin_time: int = None, end_time: int = None, line_width: float = 1.5, font_size: int = 16, color="green", figure_size: tuple = (10, 5)):
    """plot raw data.

    Args:
        data (np.array): raw data with shape [num_time_slices, num_time_series, num_features].
        selected_node_id (int): selected time series.
        begin_time (int, optional): begin time. Defaults to None.
        end_time (int, optional): end time. Defaults to None.
        line_width (float, optional): line width. Defaults to 1.5.
        font_size (int, optional): font size. Defaults to 16.
        color (str, optional): color. Defaults to "green".
        figure_size (tuple, optional): figure size. Defaults to (10, 5).
    """
    time_span = data.shape[0]
    assert begin_time < end_time, "begin_time should be less than end_time"
    assert begin_time >= 0, "begin_time should be greater than or equal to 0"
    assert end_time <= time_span, "end_time should be less than or equal to {0}".format(time_span)
    plt.rcParams['figure.figsize'] = figure_size
    plot_data = data[begin_time:end_time, selected_node_id, 0]
    plot_index = np.arange(plot_data.shape[0])
    plt.plot(plot_index, plot_data, linewidth=line_width, color=color, label="raw data")
    plt.grid()
    plt.legend(fontsize=font_size)
    # plt.savefig('vis.eps',dpi=600,format='eps', transparent=True)
    plt.savefig('Distinct_patterns_{0}.pdf'.format(DATA_NAME), dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

plot_raw_data(data=x_data, selected_node_id=5, begin_time=11000, end_time=14000, line_width=1.5, font_size=16,
              color="#000000", figure_size=(15, 5))

