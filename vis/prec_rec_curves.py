import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

# Set seaborn style
sns.set(style='whitegrid', font_scale=1.2)


rc('text', usetex=False)
sns.set(font_scale = 1.3)  # adjust so it roughly matches the caption font size

# # Manually enter precision and recall at different thresholds (0.1 to 0.9)
# # Each row: [threshold, precision, recall]
# threshold_data = torch.tensor([
#     [0.3, 0.4930, 0.7744],
#     [0.4, 0.5548, 0.7024],
#     [0.5, 0.6039, 0.5768],
#     [0.6, 0.6446, 0.4969],
#     [0.7, 0.7217, 0.3529],
#     [0.8, 0.8252, 0.2239]
# ])

# # Optional non-thresholded method (e.g. a rule-based or learned system)
# # Format: [precision, recall]
# other_method = torch.tensor([0.6889, 0.4888])

# other_method_2 = torch.tensor([0.6521, 0.3802])

# # Extract components
# thresholds = threshold_data[:, 0]
# precisions = threshold_data[:, 1]
# recalls = threshold_data[:, 2]

# # Compact plot
# plt.figure(figsize=(4.5, 4))
# plt.plot(recalls, precisions, '-o', color='blue', label='Patch Thresholding')

# # Label each threshold
# for i, t in enumerate(thresholds):
#     plt.text(recalls[i] - 0.015, precisions[i] + 0.015, f"{t.item():.1f}", fontsize=8)

# # Other method
# plt.plot(other_method[1], other_method[0], 's', markersize=8, color='red', label='Spatial Aggregation Classifier')
# plt.plot(other_method_2[1], other_method_2[0], 's', markersize=8, color='black', label='Whole Image Classifier')

# # Labels and layout
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision–Recall Curve', fontsize=10)
# plt.legend(fontsize=7, loc='lower left')
# plt.tight_layout()
# plt.savefig('outputs/figures/precision_recall_compact.png', dpi=300)

# Manually enter precision and recall at different thresholds (0.1 to 0.9)
# Each row: [threshold, precision, recall]
threshold_data = torch.tensor([
    [0.3, 0.4930, 0.7744],
    [0.4, 0.5548, 0.7024],
    [0.5, 0.6039, 0.5768],
    [0.6, 0.6446, 0.4969],
    [0.7, 0.7217, 0.3529],
    [0.8, 0.8252, 0.2239]
])

# Optional non-thresholded method (e.g. a rule-based or learned system)
# Format: [precision, recall]
threshold_data_2 = torch.tensor([    
    [0.3, 0.59, 0.67],
    [0.4, 0.62, 0.604],
    [0.5, 0.65, 0.546],
    [0.6, 0.67, 0.468],
    [0.7, 0.7, 0.387],
    [0.8, 0.734, 0.298],
])

other_method = torch.tensor([0.6521, 0.3802])

## Patch Threshold Method ##
# Extract components
thresholds = threshold_data[:, 0]
precisions = threshold_data[:, 1]
recalls = threshold_data[:, 2]

# Compact plot
plt.figure(figsize=(4.5, 4))
plt.plot(recalls, precisions, '-o', color='blue', label='Patch Thresholding')

# Label each threshold
for i, t in enumerate(thresholds):
    plt.text(recalls[i] - 0.015, precisions[i] + 0.015, f"{t.item():.1f}", fontsize=18)

## MLP Threshold Method ##
# Extract components
thresholds = threshold_data_2[:, 0]
precisions = threshold_data_2[:, 1]
recalls = threshold_data_2[:, 2]

plt.plot(recalls, precisions, '-o', color='red', label='Spatial Aggregation Classifier')

# Label each threshold
for i, t in enumerate(thresholds):
    plt.text(recalls[i] - 0.015, precisions[i] + 0.015, f"{t.item():.1f}", fontsize=18)

# Other method
plt.plot(other_method[1], other_method[0], 's', markersize=8, color='black', label='Whole Image Classifier')

# Labels and layout
plt.xlabel('Recall', fontsize=26)
plt.ylabel('Precision', fontsize=26)
plt.tick_params(axis='both', labelsize=20)
plt.legend(fontsize=30, loc='lower left')
#plt.tight_layout()
plt.show()
plt.savefig('outputs/figures/precision_recall_v4.pdf', dpi=300)
