import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, neighbors, tree, neural_network, svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline


dane = np.genfromtxt("147963-regression.txt", skip_header=1)
x_axis = dane[:, :-1]
y_axis = dane[:, -1]
print("X shape:", x_axis.shape, "Y shape:", y_axis.shape)

num_attributes = dane.shape[1]
print("Number of attributes:", num_attributes)

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

print("Number of attribute pairs with high correlation:", num_high_corr_pairs)
print("Number of attribute pairs with low correlation:", num_low_corr_pairs)

# Definition of regresors
regresors = {
    "Linear Regression": linear_model.LinearRegression(),
    "KNeighborsRegressor": neighbors.KNeighborsRegressor(),
    "DecisionTreeRegressor": tree.DecisionTreeRegressor(max_depth=2),
    "MLPRegressor": neural_network.MLPRegressor(),
    "SVR Linear Kernel": svm.SVR(kernel='linear'),
    "SVR RBF Kernel": svm.SVR(kernel='rbf')
}

# Function to calculate the difference between the values of the metrics before and after standardisation
def calculate_difference(metric, before, after):
    difference = {}
    for name in before.keys():
        difference[name] = after[name] - before[name]
    return difference

# Function to perform calculations and generate plots
def calculate_and_plot(regresors, x_axis, y_axis):
    # Calculation of metrics results without scaling
    results_mae_no_scaling = {}
    results_r2_no_scaling = {}

    for name, regresor in regresors.items():
        regresor.fit(x_axis, y_axis)
        y_pred = regresor.predict(x_axis)
        mae = mean_absolute_error(y_axis, y_pred)
        r2 = r2_score(y_axis, y_pred)
        results_mae_no_scaling[name] = mae
        results_r2_no_scaling[name] = r2

    # Calculation of metrics results with scaling
    results_mae_with_scaling = {}
    results_r2_with_scaling = {}

    for name, regresor in regresors.items():
        if name != "DecisionTreeRegressor":  # The decision tree does not need to be standardised
            regresor = make_pipeline(StandardScaler(), regresor)
        regresor.fit(x_axis, y_axis)
        y_pred = regresor.predict(x_axis)
        mae = mean_absolute_error(y_axis, y_pred)
        r2 = r2_score(y_axis, y_pred)
        results_mae_with_scaling[name] = mae
        results_r2_with_scaling[name] = r2

    # Comparison of metrics results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(results_mae_no_scaling)), list(results_mae_no_scaling.values()), align='center', alpha=0.5, label='Without Scaling')
    plt.bar(range(len(results_mae_with_scaling)), list(results_mae_with_scaling.values()), align='center', alpha=0.5, label='With Scaling')
    plt.xticks(range(len(regresors)), list(regresors.keys()), rotation=45)
    plt.title('Mean Absolute Error')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(range(len(results_r2_no_scaling)), list(results_r2_no_scaling.values()), align='center', alpha=0.5, label='Without Scaling')
    plt.bar(range(len(results_r2_with_scaling)), list(results_r2_with_scaling.values()), align='center', alpha=0.5, label='With Scaling')
    plt.xticks(range(len(regresors)), list(regresors.keys()), rotation=45)
    plt.title('R^2 Score')
    plt.ylabel('R^2')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculation of differences between values of metrics before and after standardisation
    mae_difference = calculate_difference("Mean Absolute Error", results_mae_no_scaling, results_mae_with_scaling)
    r2_difference = calculate_difference("R^2 Score", results_r2_no_scaling, results_r2_with_scaling)

    # Comparison of metrics results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(mae_difference)), list(mae_difference.values()), align='center', alpha=0.5)
    plt.xticks(range(len(regresors)), list(regresors.keys()), rotation=45)
    plt.title('Mean Absolute Error Difference')
    plt.ylabel('Difference')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(r2_difference)), list(r2_difference.values()), align='center', alpha=0.5)
    plt.xticks(range(len(regresors)), list(regresors.keys()), rotation=45)
    plt.title('R^2 Score Difference')
    plt.ylabel('Difference')

    plt.tight_layout()
    plt.show()

calculate_and_plot(regresors, x_axis, y_axis)
