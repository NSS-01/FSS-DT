class Dataset:
    def __init__(self, n, F, T, K):
        self.n = n
        self.F = F
        self.T = T
        self.K = K

# 注意所有实例都使用正确的类名 Dataset
datasets = {
    "iris": Dataset(n=150, F=13, T=113, K=3),
    "Breast Cancer": Dataset(n=699, F=9, T=7033, K=3),
    "Heart Disease": Dataset(n=303, F=13, T=375, K=3),
    "Bank Marketing": Dataset(n=45211, F=16, T=4336, K=2),
    "MNIST": Dataset(n=70000, F=784, T=2, K=2)
}

l = 64
# 遍历字典时，使用 .items()
for k, v in datasets.items():
    n, F, T, K = v.n, v.F, v.T, v.K
    # 在计算公式中使用 K 而不是 k，且修正了字符串格式化的语法
    calculation_result = 15 * (n * T * 19 * l) + l + n * l + 16 * (n * K * 11 * l)
    print(f"{k}: {calculation_result/(1204*1024*1024)}")

for k, v in datasets.items():
    n, F, T, K = v.n, v.F, v.T, v.K
    # 在计算公式中使用 K 而不是 k，且修正了字符串格式化的语法
    calculation_result = 15 * (n * T * (19*4/3) * l+2*15147) + l + n * l + 16 * (n * K * (11*4/3) * l)
    print(f"{k}: {calculation_result/(1204*1024*1024)}")


import pandas as pd


