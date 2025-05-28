import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import wasserstein_distance

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
original_data = df.values

# Privacy levels (more noise = more privacy, less utility)
privacy_levels = [0.1, 0.5, 1.0]  # standard deviation of Gaussian noise

utility_scores = []
privacy_scores = []


def compute_snr(original, noise_std):
    signal_power = np.mean(original**2)
    noise_power = noise_std**2
    return signal_power / noise_power


# iris has 4 column. just to be able to plot our data, we choose two columns


# Set fixed limits based on full dataset
# the following 7 lines are just to fix the plot limits
x_values = original_data[:, 2]  # 2 and 3 indicates specific columns
y_values = original_data[:, 3]  #  2--> Petal length, 3 -->  Petal width

# Add synthetic projections using max noise for safe margin
max_noise_std = max(privacy_levels)
x_values = np.concatenate(
    [x_values, x_values + 3 * max_noise_std, x_values - 3 * max_noise_std]
)
y_values = np.concatenate(
    [y_values, y_values + 3 * max_noise_std, y_values - 3 * max_noise_std]
)

xlim = (x_values.min(), x_values.max())
ylim = (y_values.min(), y_values.max())

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for i, noise_std in enumerate(privacy_levels):
    noise = np.random.normal(0, noise_std, size=original_data.shape)
    synthetic_data = original_data + noise

    # Metrics
    wasserstein_scores = [
        wasserstein_distance(original_data[:, j], synthetic_data[:, j])
        for j in range(original_data.shape[1])
    ]
    avg_wd = np.mean(wasserstein_scores)
    snr = compute_snr(original_data, noise_std)
    utility_scores.append(avg_wd)
    privacy_scores.append(snr)

    sns.kdeplot(
        x=original_data[:, 0],
        y=original_data[:, 3],
        fill=True,
        ax=axs[i],
        alpha=0.5,
        color="blue",
    )
    sns.kdeplot(
        x=synthetic_data[:, 0],
        y=synthetic_data[:, 3],
        fill=True,
        ax=axs[i],
        alpha=0.5,
        color="orange",
    )


    axs[i].set_xlim(xlim)
    axs[i].set_ylim(ylim)

    axs[i].set_title(
        f"Step {i+1}: noise std={noise_std:.2f}\nUtility (WD)={avg_wd:.3f}, SNR={snr:.2f}"
    )



    legend_elements = [
        Patch(facecolor="blue", alpha=0.5, label="Real"),
        Patch(facecolor="orange", alpha=0.5, label="Synthetic"),
    ]
    axs[i].legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.show()


for i, (u, p) in enumerate(zip(utility_scores, privacy_scores)):
    print(
        f"Step {i+1} | Utility (Wasserstein Distance): {u:.4f} | Privacy (SNR): {p:.2f}"
    )
