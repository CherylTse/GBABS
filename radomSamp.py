import numpy as np


def main(data,samplingrR):
    sample_size = int(samplingrR * data.shape[0])  # 采样数量
    indices = np.random.choice(data.shape[0], sample_size, replace=False)  # 随机采样的索引
    sampled_data = data[indices]  # 采样得到的数据

    return sampled_data
