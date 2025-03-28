import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
class MovingAvg(nn.Module):
    """
    利用移动平均捕捉时间序列的平滑趋势
    """
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 假设 x 的形状为 (batch, time, num_nodes)
        padding_size = (self.kernel_size - 1) // 2
        if self.kernel_size % 2 == 0:
            padding = (padding_size, padding_size + 1)
        else:
            padding = (padding_size, padding_size)
        front = x[:, 0:1, :].repeat(1, padding[0], 1)
        end = x[:, -1:, :].repeat(1, padding[1], 1)
        x_padded = torch.cat([front, x, end], dim=1)
        # 注意：AvgPool1d要求通道维度在第二个维度，因此需要变换
        x_avg = self.avg(x_padded.permute(0, 2, 1))
        x_avg = x_avg.permute(0, 2, 1)
        return x_avg

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size, hidden_size, alpha=0.5, num_iterations=2):
        """
        kernel_size: 移动平均核大小
        alpha: 消息传递中的扩散步长
        num_iterations: 有向消息传递的迭代次数
        """
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)
        self.alpha = alpha
        self.num_iterations = num_iterations

    def forward(self, x, edge_index, num_nodes=8):
        """
        x: 嵌入后的数据，形状 [B, T, hidden_size]，其中 hidden_size = num_nodes * d
        edge_index: 形状 [2, num_edges]，第一行为起始节点，第二行为终止节点（原始特征索引）
        num_nodes: 原始特征（节点）数量
        """
        B, T, hidden_size = x.shape
        d = hidden_size // num_nodes  # 每个节点的嵌入维度
        # 将 x reshape 为 [B, T, num_nodes, d]
        x_nodes = x.view(B, T, num_nodes, d)

        # 对每个节点的时间序列分别做移动平均：
        # 先合并 B 和 num_nodes 维度，使输入形状变为 [B*num_nodes, T, d]
        x_nodes_flat = x_nodes.transpose(1, 2).reshape(B * num_nodes, T, d)
        trend_time_flat = self.moving_avg(x_nodes_flat)
        # 还原为 [B, num_nodes, T, d] 再转置为 [B, T, num_nodes, d]
        trend_time = trend_time_flat.reshape(B, num_nodes, T, d).transpose(1, 2)
        fused_trend_flat=[]
        for batch_idx in range(B):
            # 构造有向邻接矩阵 A，根据 edge_index（原始特征之间的关系）
            A = torch.zeros((num_nodes, num_nodes), device=x.device)
            A[edge_index[batch_idx][0], edge_index[batch_idx][1]] = 1.0  # 保留方向：从 source 到 target
            # 按节点的入度进行归一化
            in_degree = A.sum(dim=0)  # 形状 [num_nodes]
            in_degree[in_degree == 0] = 1.0  # 防止除零
            A_norm = A / in_degree  # 归一化后的 A

            # 在节点维度上进行有向消息传递（保留方向信息）
            H = trend_time[batch_idx]  # [B, T, num_nodes, d]
            for _ in range(self.num_iterations):
                H = (1 - self.alpha) * H + self.alpha * torch.einsum("ji,tjd->tid", A_norm, H)

            fused_trend = H  # [B, T, num_nodes, d]
            # 如果需要与原始嵌入做比较，可 reshape 回 [B, T, hidden_size]
            fused_trend_flat.append(fused_trend.contiguous().view(T, hidden_size))
        # 计算残差：原始嵌入 x 与融合趋势之间的差异
        fused_trend_flat=torch.stack(fused_trend_flat, dim=0)
        residual = x - fused_trend_flat
        return residual, fused_trend_flat
