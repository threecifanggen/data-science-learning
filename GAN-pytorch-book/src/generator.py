import torch
import random

generate_real = lambda: torch.FloatTensor([
    random.uniform(0.8, 1),
    random.uniform(0.0, 0.2),
    random.uniform(0.8, 1),
    random.uniform(0.0, 0.2),
])
generate_real.__doc__ = "生成真实数据"


generate_random = lambda size: torch.rand(size)
generate_random.__doc__ = "生成随机噪音"

generate_random_image = lambda size: torch.rand(size)
generate_random_seed = lambda size: torch.randn(size)