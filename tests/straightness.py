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

fig, axes = plt.subplots(2, 1)
axes[0].plot(np.array(straightness) / 100.)
axes[1].plot(np.cumsum(straightness) / 100.)
pyplot.tight_layout()
plt.show()

z = np.cumsum(straightness)

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
    for i in range(1, N):
        needle = d * i
        idx = np.abs(straightness_cum - needle).argmin()
        ts.append(idx / len(straightness_cum))

    ts = [0] + ts + [1.]
    return ts

for i in range(2, 11):
    print(f"    {i}:", find_piecewise_time(z, i))
