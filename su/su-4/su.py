import numpy as np
import matplotlib.pyplot as plt

dane = np.genfromtxt("147963-regression.txt", skip_header=1)
x_axis = dane[:, :-1]
y_axis = dane[:, -1]
print("Wymiar x", x_axis.shape, "Wymair y", y_axis.shape)

num_attributes = dane.shape[1]
print("Liczba atrybutów:", num_attributes)

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

correlation_matrix = np.corrcoef(dane, rowvar=False)

# High corelation (np. 0.7)
high_corr_threshold = 0.7

# Low corelation (e.g. 0.1)
low_corr_threshold = 0.1

high_corr_pairs = np.where(np.abs(correlation_matrix) > high_corr_threshold)
high_corr_pairs = [(i, j) for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]) if i != j]
num_high_corr_pairs = len(high_corr_pairs)

low_corr_pairs = np.where(np.abs(correlation_matrix) < low_corr_threshold)
low_corr_pairs = [(i, j) for i, j in zip(low_corr_pairs[0], low_corr_pairs[1]) if i != j]
num_low_corr_pairs = len(low_corr_pairs)

print("Liczba par atrybutów z wysoką korelacją:", num_high_corr_pairs)
print("Liczba par atrybutów z niską korelacją:", num_low_corr_pairs)