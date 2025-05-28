import torch
import torch.nn as nn
import opt
args = opt.opt()

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        if args.net == 'LWENet':
            input_dim = 256*3
        # 定义全连接层
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 从输入维度512到隐藏层
            nn.ReLU(),  # 使用ReLU激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 从隐藏层256到下一个隐藏层
            nn.ReLU(),
            # nn.Linear(hidden_dim, 2),  # 输出为一个值，用于二分类判别
            # nn.Softmax(-1)  # 使用Sigmoid激活函数将输出限制在[0, 1]
            nn.Linear(hidden_dim, 1),  # 输出为一个值，用于二分类判别
            nn.Sigmoid()  # 使用Sigmoid激活函数将输出限制在[0, 1]
        )

    def forward(self, x):
        return self.model(x)


# 示例输入
if __name__ == "__main__":
    # 实例化域判别器
    domain_discriminator = DomainDiscriminator()

    # 输入为 [b, 512] 维特征向量, 例如 b = 8
    x = torch.randn((8, 512))  # 8个样本，每个样本512维特征
    output = domain_discriminator(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output}")  # 输出为 [b, 1]，每个样本的二分类结果
