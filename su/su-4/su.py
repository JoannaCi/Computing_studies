import numpy as np
import matplotlib.pyplot as plt

dane = np.genfromtxt("147963-regression.txt", skip_header=1)
x_axis = dane[:, :-1]
y_axis = dane[:, -1]

# Plot
plt.figure(figsize=(12, 8))
bp = plt.boxplot(x_axis)

num_boxes = len(bp['boxes'])
colors = plt.cm.rainbow(np.linspace(0, 1, num_boxes))

for box, color in zip(bp['boxes'], colors):
    box.set(color=color, linewidth=2)

plt.xticks(np.arange(1, x_axis.shape[1] + 1), rotation=90)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.title('Boxplot of Features', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Histogram
plt.figure(figsize=(8, 6))
counts, bins, _ = plt.hist(x_axis, bins=20, edgecolor='black', alpha=0.7)

colors = np.random.rand(len(counts))  # Random colours

for i in range(len(bins) - 1):  # Iterate up to len(bins) - 1, so there won't be index out of range
    plt.bar(bins[i], counts[i], width=bins[i + 1] - bins[i], color=plt.cm.viridis(colors[i]), edgecolor='black', alpha=0.7)

plt.xlabel('Class', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Class', fontsize=16)

plt.grid(True, linestyle='--', alpha=0.7)
plt.show()