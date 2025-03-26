# Description: 计算硬件的TFLOPS
# RTX 3060：
#   27 TF16LOPS
#   12.6 TF32LOPS
#   0.18 TF64LOPS
import torch
from torch.utils import benchmark

typ = torch.float32  #数据精度
n = 1024 * 16
a = torch.randn(n, n).type(typ).cuda()
b = torch.randn(n, n).type(typ).cuda()
t= benchmark.Timer(
      stmt='a @ b',
      globals={'a': a, 'b': b})
x = t.timeit(50)
print(2*n**3 / x.median /1e12)

