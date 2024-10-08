import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

from config import EPOCH, LR, DOWNLOAD_MNIST
from datapreprocessing.data import load_data, get_train_test_loaders
from models.cnn import CNN
from models.mlp import MLP

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 加载数据
train_data, test_data = load_data(DOWNLOAD_MNIST)

# 获取数据加载器
train_loader, test_loader = get_train_test_loaders(train_data, test_data, batch_size=64)
#train_loader：返回每个批次的数据，形状为 (64, 1, 28, 28)

# 进行测试
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_x = test_x.view(test_x.size(0), -1)  # 展平每个图像
test_y = test_data.targets[:2000]

"""
test_data.data：原始测试数据，假设它是一个大的张量，形状为 [2000, 1, 28, 28]（对于 MNIST 数据集）。
torch.unsqueeze(test_data.data, dim=1)：在第二个维度上增加一个大小为1的维度，形状变为 [2000, 1, 1, 28, 28]。
.type(torch.FloatTensor)：转换数据类型为浮点型，形状不变。
[:2000]：选择前2000个样本，形状为 [2000, 1, 1, 28, 28]。
test_x = test_x.view(test_x.size(0), -1)：展平每个图像，形状变为 [2000, 784]。
"""

# 初始化模型
mlp = MLP(28*28, 50, 10)  # 修改输入大小为 28*28

# 训练
optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(b_x.size(0), -1) / 255  # 展平图像并归一化展平图像并归一化，形状变为 [64, 784]。
        b_y = b_y.long()  # 确保 b_y 是长整型
        output = mlp(b_x)#模型前向传播，输出形状为 [64, 10]（假设有10个类别）。
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = mlp(test_x)#模型前向传播，输出形状为 [2000, 10]。
            pred_y = torch.max(test_output, 1)[1].data.numpy()#获取预测类别的索引，形状为 [2000]。
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# 保存模型
torch.save(mlp.state_dict(), 'result/mlp.pkl')

# 加载模型
model_path = 'result/mlp.pkl'
if os.path.exists(model_path):
    mlp.load_state_dict(torch.load(model_path))
    mlp.eval()
    print("Model loaded successfully.")
else:
    print("Model file not found.")

# 打印 10 个预测结果
inputs = test_x[:10]
test_output = mlp(inputs)
pred_y = torch.max(test_output, 1)[1].data.numpy()

# 重新形状化图像
images = inputs.view(-1, 28, 28)  # -1 让 PyTorch 自动计算批次大小

# 可视化预测结果
fig, axes = plt.subplots(1, 10, figsize=(12, 2))  # 假设你要显示 10 个图像
for i, ax in enumerate(axes):
    ax.imshow(images[i].numpy(), cmap='gray')  # 使用 .numpy() 转换为 NumPy 数组
    ax.title.set_text(f'Pred: {pred_y[i]}')
    ax.axis('off')

plt.show()