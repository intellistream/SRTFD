import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2, hidden_dim=128, num_heads=4):
        super(TransformerFeatureExtractor, self).__init__()

        self.encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        self.fc = nn.Linear(input_dim, 10)

    def forward(self, x):
        
        x = x.float().unsqueeze(0)  # 添加一个维度作为序列长度 (1, batch_size, dim)
        encoded = self.encoder(x)  # 进行编码
        encoded = encoded.squeeze(0)  # 移除序列长度维度
       
        features = self.fc(encoded)  # 线性变换
        return F.softmax(features, dim=1)  # 返回 softmax 后的结果
    
# input_dim = 1000
# output_dim = 10
# batch_size = 3
# num_batches = 100
# num_epochs = 5

# # 初始化模型
# model = TransformerFeatureExtractor(input_dim, output_dim)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# # 模拟训练过程
# for epoch in range(num_epochs):
#     for batch_idx in range(num_batches):
#         # 生成随机输入数据和标签
#         x_batch = torch.randn(batch_size, input_dim)
#         y_batch = torch.randint(0, output_dim, (batch_size,))
        
#         # 前向传播
#         optimizer.zero_grad()
#         outputs = model(x_batch)
        
#         # 计算损失
#         loss = criterion(outputs, y_batch).requires_grad_(True)
        
#         # 反向传播和优化
#         loss.backward()
#         optimizer.step()
        

# print('Finished Training')