import torch
import soundfile
import numpy as np

from matplotlib import pyplot as plt
from sympy.physics.quantum.circuitplot import pyplot

# th = "/Users/liupeng/Downloads/LJSpeech-distill-reverse/sample_0000003_008.th"
# data = torch.load(th)
# print(data['z0'].mean(), data['z0'].std())
# soundfile.write('test_reverse_noise.wav', data['z0'].numpy(), 22050, 'PCM_16')

th = "/home/feanorliu/PycharmProjects/rfwave/pred.th"
d = torch.load(th, map_location='cpu')
straightness = []
v_final = d[-1]
for v in d[:-1]:
    diff = (v - v_final).view(v.size(0), -1)
    straight = torch.norm(diff, p='fro', dim=1, keepdim=False)
    straightness.append(straight.mean())

np.save('straightness.npy', straightness)
z = np.cumsum(straightness) / 100.

half_value = z[-1] / 2
one_third_value = z[-1] / 3
two_third_value = 2 * z[-1] / 3

index_half = (np.abs(z - half_value)).argmin()
index_one_third = (np.abs(z - one_third_value)).argmin()
index_two_third = (np.abs(z - two_third_value)).argmin()

print("Index for 1/2 of the total sum:", index_half)
print("Index for 1/3 of the total sum:", index_one_third)
print("Index for 2/3 of the total sum:", index_two_third)
def find_piecewise_time(straightness_cum, N):
    d = straightness_cum[-1] / N
    ts = []
    neddles = [0.]
    for i in range(1, N):
        needle = d * i
        idx = np.abs(straightness_cum - needle).argmin()
        ts.append(idx / len(straightness_cum))
        neddles.append(straightness_cum[idx])

    ts = [0] + ts + [1.]
    return ts, neddles

for i in range(2, 11):
    ts, neddles = find_piecewise_time(z, i)
    print(f"    {i}:", ts)


print(ts)
print(neddles)

fig, axes = plt.subplots(2, 1)
x = np.linspace(0, 1, 101)[:-1]
axes[0].plot(x, np.array(straightness) / 100.)
axes[1].plot(x, np.cumsum(straightness) / 100.)

for x, y in zip([0] + ts, [0] + neddles):
    axes[1].plot(x, y, 'rx')
    axes[1].vlines(x=x, ymin=0, ymax=y, color='gray', linestyle='--', linewidth=0.8)  # 垂直于x轴的线
    axes[1].hlines(y=y, xmin=0, xmax=x, color='gray', linestyle='--', linewidth=0.8)  # 垂直于y轴的线
    # plt.text(x, 0, f'{x:.2f}', fontsize=9, ha='center', va='top')  # 在x轴上标注x值
    # plt.text(0, y, f'{y:.2f}', fontsize=9, ha='right', va='center')  # 在y轴上标注y值

# 设置 x 轴范围
axes[0].set_xlim([0, 1])
axes[1].set_xlim([0, 1])
# 设置 y 轴范围
axes[0].set_ylim([min(straightness) / 100., max(straightness) / 100.])
axes[1].set_ylim([0, max(np.cumsum(straightness) / 100.)])
pyplot.tight_layout()
plt.show()
