# Description: 测试ROSbag文件的读取
# Author: Qikai Li
# Dependency: rosbag, pathlib, pycryptodome

import rosbag
from pathlib import Path

# 使用 matplotlib 可视化数据
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import torch


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


# 获取当前文件夹路径
current_path = Path(__file__).parent
print(current_path)
# 打开ROSbag文件
bag = rosbag.Bag(current_path / "data" / "scalar-0.3-01.bag")
print(bag)

# get topic /joint_posvel
motor_vel_data = []
motor_pos_data = []
angular_vel_x_data = []
angular_vel_y_data = []
angular_vel_z_data = []
quat_data = []

for topic, msg, t in bag.read_messages(topics=["/joint_posvel"]):
    motor_pos_data.append(msg.data[0])
    motor_vel_data.append(msg.data[10])
    if len(motor_vel_data) >= 1500:
        break

for topic, msg, t in bag.read_messages(topics=["/body_twist"]):
    angular_vel_x_data.append(msg.angular.x)
    angular_vel_y_data.append(msg.angular.y)
    angular_vel_z_data.append(msg.angular.z)
    if len(angular_vel_x_data) >= 1500:
        break

for topic, msg, t in bag.read_messages(topics=["/body_pose"]):
    quat_data.append([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    if len(quat_data) >= 1200:
        break
quat_tensor = torch.tensor(quat_data)

gravity = torch.tensor([0.0, 0.0, -9.81]).repeat(quat_tensor.shape[0], 1)
proj_grav_tensor = quat_rotate_inverse(quat_tensor, gravity)

# print(motor_vel_data)
# print(len(motor_vel_data))

# 绘制电机速度数据
# plt.plot(motor_vel_data)
# plt.draw()

# 绘制电机位置数据
# plt.plot(motor_pos_data)
# plt.draw()

# 绘制角速度数据
# plt.plot(angular_vel_x_data)
# plt.plot(angular_vel_y_data)
# plt.plot(angular_vel_z_data)
# plt.draw()

# 绘制重力投影
# plt.plot(proj_grav_tensor[:, 0].numpy())
# plt.plot(proj_grav_tensor[:, 1].numpy())
# plt.plot(proj_grav_tensor[:, 2].numpy())
# plt.draw()


from scipy import stats

# noise_data = motor_vel_data
# noise_data = motor_pos_data
# noise_data = angular_vel_x_data
# noise_data = yaw_data
# noise_data = proj_grav_tensor[:, 1].numpy()
noise_data = np.array(quat_data)[:, 3]

# 拟合正态分布
mu, sigma = stats.norm.fit(noise_data)

print(f"Estimated mean (mu): {mu}")
print(f"Estimated standard deviation (sigma): {sigma}")

# 绘制拟合曲线
plt.figure(figsize=(10, 6))

# 绘制原始数据的直方图
sns.histplot(noise_data, kde=False, stat="density", bins=30, label="Data")
# plt.title("Histogram of Noise Data")
# plt.xlabel("Noise Value")
# plt.ylabel("Density")
# plt.show()
# 绘制拟合的正态分布曲线
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, "k", linewidth=2, label="Fitted normal distribution")

plt.title("Fitted Normal Distribution")
plt.xlabel("Noise Value")
plt.ylabel("Density")
plt.legend()
plt.show()
