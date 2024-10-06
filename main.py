import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from config import EPOCH, LR, DOWNLOAD_MNIST
from datapreprocessing.data import load_data, get_train_test_loaders
from models.cnn import CNN

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 加载数据
train_data, test_data = load_data(DOWNLOAD_MNIST)

# 获取数据加载器
train_loader, test_loader = get_train_test_loaders(train_data,test_data, batch_size=64)

# 进行测试
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.targets[:2000]

cnn = CNN()
print(cnn)

# 训练
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad() #梯度清零
        loss.backward() # 前向传播
        optimizer.step()
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# 保存模型
torch.save(cnn.state_dict(), 'result/cnn2.pkl')

# 加载模型
model_path = 'result/cnn2.pkl'
if os.path.exists(model_path):
    cnn.load_state_dict(torch.load(model_path))
    cnn.eval()
    print("Model loaded successfully.")
else:
    print("Model file not found.")

# 打印 10 个预测结果
inputs = test_x[:10]
test_output = cnn(inputs)
pred_y = torch.max(test_output, 1)[1].data.numpy()

# 可视化预测结果
fig, axes = plt.subplots(1, 8, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(inputs[i].squeeze().numpy(), cmap='gray')
    ax.title.set_text(f'Pred: {pred_y[i]}')
    ax.axis('off')

plt.show()