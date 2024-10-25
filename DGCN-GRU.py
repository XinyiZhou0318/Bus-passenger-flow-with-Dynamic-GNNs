import pandas as pd
import networkx as nx
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 文件路径
stops_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\stops.csv"
stop_times_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\stop_times.csv"
zero_passenger_stops_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\zero_passenger_stops.csv"
arrival_data_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\arrival_by_time_filtered.csv"
true_values_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\true_values_17_00.xlsx"
save_folder = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\Dynamic_Graphs"
results_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\prediction_results_DGCN-GRU.xlsx"
loss_excel_path = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\loss_values_DGCN-GRU.xlsx"
input_data_path_1 = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\input_data_1_DGCN-GRU.xlsx"
input_data_path_2 = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\input_data_2_DGCN-GRU.xlsx"
input_data_path_3 = r"C:\Users\86176\Desktop\GNNs &Transportation\test\Dynamic GNN\input_data_3_DGCN-GRU.xlsx"

# 创建保存文件夹（如果不存在）
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 读取CSV文件
stops_df = pd.read_csv(stops_path)
stop_times_df = pd.read_csv(stop_times_path)
zero_passenger_stops_df = pd.read_csv(zero_passenger_stops_path)
arrival_df = pd.read_csv(arrival_data_path)
true_values_df = pd.read_excel(true_values_path)

# 将 zero_passenger_stops_df 与 stop_times_df 在 'stop' 和 'stop_id' 上合并
zero_passenger_stops_df = pd.merge(zero_passenger_stops_df, stop_times_df[['stop_id', 'trip_id']], left_on='stop', right_on='stop_id', how='left')

# 获取所有的时间列并检查
time_columns = arrival_df.columns[1:]
print(f"Available time columns: {time_columns}")

# 目标日期和时间段
target_date = "2022-01-30"
start_time = "15:00:00"
end_time = "16:45:00"

# 筛选目标时间段数据
target_columns = [col for col in time_columns if target_date in col and start_time <= col.split(' ')[1] <= end_time]
print(f"Selected time columns for {target_date} between {start_time} and {end_time}: {target_columns}")

# 准备其他两组输入的时间列
input_dates_2 = ["2022-01-26", "2022-01-27", "2022-01-29"]
input_dates_3 = ["2022-01-17", "2022-01-23", "2022-01-24"]

input_columns_2 = [col for col in time_columns if any(date in col for date in input_dates_2) and "17:00:00" in col]
input_columns_3 = [col for col in time_columns if any(date in col for date in input_dates_3) and "17:00:00" in col]

# 只保留 stop_times.csv 中的 stop_id
stop_times_ids = stop_times_df['stop_id'].unique()
filtered_stops_df = stops_df[stops_df['stop_id'].isin(stop_times_ids)]

# 创建初始有向图
G_original = nx.DiGraph()

# 添加节点
for _, row in filtered_stops_df.iterrows():
    stop_id = row['stop_id']
    stop_name = row['stop_name']
    G_original.add_node(stop_id, stop_name=stop_name)

# 根据 trip_id 和 stop_sequence 构建有向边
for trip_id in stop_times_df['trip_id'].unique():
    trip_data = stop_times_df[stop_times_df['trip_id'] == trip_id].sort_values('stop_sequence')
    stop_ids = trip_data['stop_id'].tolist()
    for i in range(len(stop_ids) - 1):
        if stop_ids[i] in stop_times_ids and stop_ids[i + 1] in stop_times_ids:
            G_original.add_edge(stop_ids[i], stop_ids[i + 1])

# 取消的站点的 trip_id
canceled_trips = zero_passenger_stops_df[zero_passenger_stops_df['ons'] == 0]['trip_id'].unique()

# 删除取消的站点并重连前后节点
for trip_id in canceled_trips:
    canceled_stops_trip = zero_passenger_stops_df[zero_passenger_stops_df['trip_id'] == trip_id]['stop'].tolist()

    # 判断是单个站点还是整个线路取消
    trip_stops_in_gt = stop_times_df[stop_times_df['trip_id'] == trip_id]['stop_id'].tolist()
    if set(trip_stops_in_gt) == set(canceled_stops_trip):
        # 整条线路取消，删除所有站点
        for stop_id in trip_stops_in_gt:
            if stop_id in G_original:
                G_original.remove_node(stop_id)
    else:
        # 单个站点取消，重新连接前后站点
        for stop_id in canceled_stops_trip:
            if stop_id in G_original:
                predecessors = list(G_original.predecessors(stop_id))
                successors = list(G_original.successors(stop_id))
                if predecessors and successors:
                    for pred in predecessors:
                        for succ in successors:
                            G_original.add_edge(pred, succ)
                G_original.remove_node(stop_id)

# 准备三组输入的数据
arrival_data_range_1 = arrival_df[['stop'] + target_columns]
arrival_data_range_1.columns = ['Stop_ID'] + target_columns
arrival_data_range_1 = arrival_data_range_1.set_index('Stop_ID').fillna(0)

arrival_data_range_2 = arrival_df[['stop'] + input_columns_2]
arrival_data_range_2.columns = ['Stop_ID'] + input_columns_2
arrival_data_range_2 = arrival_data_range_2.set_index('Stop_ID').fillna(0)

arrival_data_range_3 = arrival_df[['stop'] + input_columns_3]
arrival_data_range_3.columns = ['Stop_ID'] + input_columns_3
arrival_data_range_3 = arrival_data_range_3.set_index('Stop_ID').fillna(0)

# # 保存输入数据到 Excel
# arrival_data_range_1.to_excel(input_data_path_1)
# arrival_data_range_2.to_excel(input_data_path_2)
# arrival_data_range_3.to_excel(input_data_path_3)
# print(f"Input data saved to {input_data_path_1}, {input_data_path_2}, {input_data_path_3}")

# 转换为 PyTorch 张量
node_features_1 = torch.tensor(arrival_data_range_1.values, dtype=torch.float)
node_features_2 = torch.tensor(arrival_data_range_2.values, dtype=torch.float)
node_features_3 = torch.tensor(arrival_data_range_3.values, dtype=torch.float)

# 创建 PyTorch Geometric 的图数据对象
data_list_1 = []
data_list_2 = []
data_list_3 = []

for col in target_columns:
    Gt_1 = G_original.copy()
    edge_index_1 = torch.tensor(list(Gt_1.edges)).t().contiguous().long()
    valid_mask_1 = (edge_index_1[0] < len(Gt_1.nodes)) & (edge_index_1[1] < len(Gt_1.nodes))
    edge_index_1 = edge_index_1[:, valid_mask_1]
    data_1 = Data(x=torch.tensor(arrival_data_range_1[[col]].values, dtype=torch.float), edge_index=edge_index_1)
    data_list_1.append(data_1)

for col in input_columns_2:
    Gt_2 = G_original.copy()
    # 对于每一天的数据，使用不同的 Gt 副本
    edge_index_2 = torch.tensor(list(Gt_2.edges)).t().contiguous().long()
    valid_mask_2 = (edge_index_2[0] < len(Gt_2.nodes)) & (edge_index_2[1] < len(Gt_2.nodes))
    edge_index_2 = edge_index_2[:, valid_mask_2]
    data_2 = Data(x=torch.tensor(arrival_data_range_2[[col]].values, dtype=torch.float), edge_index=edge_index_2)
    data_list_2.append(data_2)

for col in input_columns_3:
    Gt_3 = G_original.copy()
    # 对于每一天的数据，使用不同的 Gt 副本
    edge_index_3 = torch.tensor(list(Gt_3.edges)).t().contiguous().long()
    valid_mask_3 = (edge_index_3[0] < len(Gt_3.nodes)) & (edge_index_3[1] < len(Gt_3.nodes))
    edge_index_3 = edge_index_3[:, valid_mask_3]
    data_3 = Data(x=torch.tensor(arrival_data_range_3[[col]].values, dtype=torch.float), edge_index=edge_index_3)
    data_list_3.append(data_3)

# 定义 DGCN-GRU 模型
class Modified_GCN_GRU(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, gru_hidden_dim, out_channels):
        super(Modified_GCN_GRU, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)  # GCN的输出维度与输入维度一致  # 确保输出维度与输入维度相同
        self.gru = torch.nn.GRU(hidden_dim, gru_hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(gru_hidden_dim, out_channels)

        # 自身特征的权重矩阵和邻居节点变化的权重矩阵
        self.W_0 = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.W_t = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # 定义 W1 和 W2 权重矩阵
        self.W1 = torch.nn.Parameter(torch.randn(3 * gru_hidden_dim, 96))
        self.W2 = torch.nn.Parameter(torch.randn(96, out_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 第一步：从邻居节点进行标准的 GCN 聚合
        deg = torch.sqrt(torch.tensor([G_original.degree(node) for node in list(G_original.nodes)], dtype=torch.float) + 1e-9)
        deg_inv_sqrt = torch.reciprocal(deg + 1e-9)  # 避免除以零
        if edge_index.size(1) > 0:
            normalization = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        else:
            normalization = torch.ones(edge_index.size(1))
        normalization = normalization if normalization.size(0) > 0 else torch.ones(edge_index.size(1))  # 确保归一化系数不为空，且大小正确
        h_neigh = self.conv1(x, edge_index) * normalization.view(-1, 1) if normalization.size(0) > 0 else self.conv1(x, edge_index)  # 邻居节点特征
        h_neigh = F.relu(h_neigh)

        # 第二步：对节点自身的特征进行变换 (W_0 项)
        h_self = torch.matmul(x, self.W_0)  # 自身特征的变换

        # 第三步：计算邻居节点的变化并进行变换 (W_t 项)
        h_change = torch.matmul(h_neigh - x, self.W_t)  # 对邻居节点变化进行变换

        # 第四步：结合所有的贡献
        h_combined = h_neigh + h_self + h_change

        # 应用激活函数 σ
        h_combined = torch.sigmoid(h_combined)

        # 第五步：应用 GRU
        h_combined = h_combined.unsqueeze(0)  # 添加批次维度
        gru_out, _ = self.gru(h_combined)
        gru_out = gru_out.squeeze(0)  # 移除批次维度

        return gru_out

input_dim = 1
hidden_dim = input_dim  # GCN的隐藏层输出维度与输入维度一致
gru_hidden_dim = 32
output_dim = 1

model = Modified_GCN_GRU(input_dim, hidden_dim, gru_hidden_dim, output_dim)

# 使用 Adam 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

# 训练模型
loss_values = []
log_loss_values = []  # 添加记录对数损失值
model.train()
for epoch in range(500):
    if epoch < 500:
        optimizer.param_groups[0]['lr'] = 1e-4  # 较大的时间步长
    else:
        optimizer.param_groups[0]['lr'] = 1e-4  # 较小的时间步长
    optimizer.zero_grad()

    # 分别计算三组数据的输出
    out_1 = torch.stack([model(data) for data in data_list_1], dim=0).mean(dim=0)
    out_2 = torch.stack([model(data) for data in data_list_2], dim=0).mean(dim=0)
    out_3 = torch.stack([model(data) for data in data_list_3], dim=0).mean(dim=0)

    # 使用公式 (11) 结合三组预测结果
    combined_input = torch.cat([out_1, out_2, out_3], dim=-1)
    combined_out = F.relu(torch.matmul(combined_input, model.W1))
    combined_out = torch.matmul(combined_out, model.W2)

    target = node_features_1[:, -1].unsqueeze(1)
    loss = loss_fn(combined_out, target)
    loss_values.append(loss.item())
    log_loss_values.append(np.log(loss.item() + 1e-10))  # 记录对数损失值，避免log(0))
    loss.backward()
    optimizer.step()

# 模型预测
model.eval()
predicted_flow_1 = torch.stack([model(data) for data in data_list_1], dim=0).mean(dim=0).detach().numpy()
predicted_flow_2 = torch.stack([model(data) for data in data_list_2], dim=0).mean(dim=0).detach().numpy()
predicted_flow_3 = torch.stack([model(data) for data in data_list_3], dim=0).mean(dim=0).detach().numpy()

# 使用公式 (11) 计算最终预测值
combined_input = np.concatenate([predicted_flow_1, predicted_flow_2, predicted_flow_3], axis=-1)
predicted_flow = np.maximum(0, np.dot(combined_input, model.W2.detach().numpy())).flatten()

# 获取当天的真实值
true_values_day = true_values_df[['Stop_ID', f'{target_date}_True_Value']].dropna()

# 确保预测值和真实值对齐，预测值的长度与真实值一致
predicted_flow = predicted_flow[:len(true_values_day)]

# 筛选1月30日取消的站点
canceled_stops_30jan = zero_passenger_stops_df[(zero_passenger_stops_df['date'] == target_date) &
                                               (zero_passenger_stops_df['ons'] == 0) &
                                               (zero_passenger_stops_df['offs'] == 0)]['stop']

# 将取消的站点的预测值设置为0
for stop_id in canceled_stops_30jan:
    idx = true_values_day.index[true_values_day['Stop_ID'] == stop_id].tolist()
    if idx:
        for i in idx:
            predicted_flow[i] = 0

# 定义真实值 true_values 和预测值 predicted_values
true_values = true_values_day[f'{target_date}_True_Value'].values
predicted_values = predicted_flow

# 计算 MAE
mae = mean_absolute_error(true_values, predicted_values)

# 计算 RMSE
rmse = np.sqrt(mean_squared_error(true_values, predicted_values))

# 计算准确率 (误差在5%以内和10%以内的正确率)
def calculate_accuracy(true_vals, pred_vals, threshold):
    # 计算百分比误差
    percent_error = np.abs(true_vals - pred_vals)
    # 计算在给定阈值内的正确预测个数
    correct_predictions = np.sum(percent_error <= threshold * true_vals)
    # 计算正确预测的占比
    accuracy = (correct_predictions / len(true_vals)) * 100
    return accuracy, correct_predictions

# 计算误差在5%以内的准确率
accuracy_5_percent, correct_count_5 = calculate_accuracy(true_values, predicted_values, 0.05)

# 计算误差在10%以内的准确率
accuracy_10_percent, correct_count_10 = calculate_accuracy(true_values, predicted_values, 0.10)

# 打印结果
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"Accuracy (<=5%): {accuracy_5_percent}%")
print(f"Accuracy (<=10%): {accuracy_10_percent}%")

# 保存预测结果
results_df = pd.DataFrame({'Stop_ID': true_values_day['Stop_ID'], 'Predicted_Value': predicted_flow, 'True_Value': true_values})
results_df.to_excel(results_path, index=False)
print(f"Prediction results for {target_date} saved to {results_path}")

# 绘制训练损失和对数损失的下降过程
plt.figure()
plt.plot(range(len(loss_values)), loss_values, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss and Log Loss over Epochs')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(log_loss_values)), log_loss_values, label='Log Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss and Log Loss over Epochs')
plt.legend()
plt.show()

# 保存损失值到Excel文件
loss_df = pd.DataFrame({
    'Epoch': range(len(loss_values)),
    'Loss': loss_values,
    'Log_Loss': log_loss_values
})
loss_df.to_excel(loss_excel_path, index=False)
print(f"Loss values saved to {loss_excel_path}")
