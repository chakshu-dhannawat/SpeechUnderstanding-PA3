
from dataloader import get_dataloader
from metrics import calculate_auc, calculate_eer
import torch
import numpy as np
from model_SSL import Model

### Loading the dataloader

DATASET_PATH = r"C:\Users\chaks\SU_PA3\Dataset_Speech_Assignment\Dataset_Speech_Assignment"
dataloader = get_dataloader(DATASET_PATH, batch_size=32, shuffle=True)


### Loading the model
MODEL_PATH = r"C:\Users\chaks\SU_PA3\Best_LA_model_for_DF.pth"
model = Model(model_path = MODEL_PATH)


### Evaluation
predictions = []
true_labels = []

# Iterate over the dataset to get predictions and true labels
for data, labels in dataloader:
    with torch.no_grad():
        outputs = model(data)
        predictions.extend(outputs.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# Calculate AUC
auc = calculate_auc(true_labels, predictions)

# Calculate ERR
err = calculate_eer(true_labels, predictions)

print("AUC:", auc)
print("ERR:", err)

