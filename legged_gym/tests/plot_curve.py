import math
import numpy as np
import matplotlib.pyplot as plt


def smooth_sqr_wave(phase):
    eps = 0.2
    p = 2.0 * math.pi * phase * 0.7  #  * self.phase_freq
    return np.sin(p) / (2 * np.sqrt(np.sin(p) ** 2.0 + (eps) ** 2.0)) + 1.0 / 2.0


def relaxed_barrier_function(c, d_lower, d_upper, delta):
    def B(z, delta):
        return np.where(z > delta, np.log(z), np.log(delta) - 0.5 * (np.square((z - 2.0 * delta) / delta) - 1.0))

    return B(-d_lower + c, delta) + B(d_upper - c, delta)


# x = np.linspace(-2 * np.pi, 2 * np.pi, 400)  # 从-2π到2π生成400个点
# y = smooth_sqr_wave(x)  # 计算正弦值

x = np.linspace(-2 * 10, 2 * 10, 40000)  # 从-2π到2π生成400个点
y = relaxed_barrier_function(x, -5.0, 10.0, 0.1)

# 绘制图像
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(x, y, label="f(x)", color="blue", linestyle="-")  # 绘制正弦曲线

# 显示图像
plt.show()
