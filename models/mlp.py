import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        # 第一层隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 第二层隐藏层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 输出层
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 通过第一层隐藏层后使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二层隐藏层后使用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 产生输出
        x = self.out(x)
        return x

"""
# 假设输入特征大小为20，隐藏层大小为50，分类类别数为10
input_size = 20
hidden_size = 50
num_classes = 10

# 创建MLP模型实例
#mlp = MLP(input_size, hidden_size, num_classes)
"""


