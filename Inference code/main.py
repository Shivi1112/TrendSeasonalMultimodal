import os
import torch
import pickle
import numpy as np
import pandas as pd
from torchvision import transforms
from models.drfuse import DrFuseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = DrFuseModel(
    hidden_size=256, num_classes=25, ehr_dropout=0.3, ehr_n_head=4, ehr_n_layers=1
)

weights = torch.load("checkpoints/best_model-f11-epoch=11_new.ckpt", map_location=device)["state_dict"]

weights = {
    k.replace("model.", ""): v
    for k, v in weights.items()
    if "model." in k
}
weights = {
    k.replace("ehr_", "ehr_model."): v
    for k, v in weights.items()
    if "ehr_" in k
}

model.load_state_dict(weights)
model.to(device).eval()

# Load data
ehr_data = pickle.load(open("phenotyping/ehr_phenotyping_48h_val.pkl", "rb"))
cxr_data = pickle.load(open("cxr_phenotyping_48h_val.pkl", "rb"))
paired_data = pickle.load(open("data/phenotyping/paired_ehr_cxr_files.pkl", "rb"))["val"]

# Transform for CXR images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLASSES = [
    "Acute and unspecified renal failure", "Acute cerebrovascular disease",
    "Acute myocardial infarction", "Cardiac dysrhythmias", "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care", "Conduction disorders",
    "Congestive heart failure; nonhypertensive", "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications", "Diabetes mellitus without complication",
    "Disorders of lipid metabolism", "Essential hypertension", "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage", "Hypertension with complications and secondary hypertension",
    "Other liver diseases", "Other lower respiratory disease", "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)", "Septicemia (except in labor)", "Shock",
]

# Pad EHR sequences
def pad_zeros(arr, min_length=None):
    dtype = arr[0].dtype
    seq_lengths = [x.shape[0] for x in arr]
    max_len = max(seq_lengths)
    padded = [
        np.concatenate(
            [x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0
        ) for x in arr
    ]
    if min_length and padded[0].shape[0] < min_length:
        padded = [
            np.concatenate(
                [x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0
            ) for x in padded
        ]
    return np.array(padded), seq_lengths

# Inference function
def predict(ehr_key, dicom_id):
    x = ehr_data[ehr_key]["data"]
    x[x > 10] = 0
    x[x < -10] = 0

    has_cxr = dicom_id is not None
    pairs = [1 if has_cxr else 0]

    if has_cxr:
        img = transform(cxr_data[dicom_id])
    else:
        img = torch.zeros(3, 224, 224)

    x, seq_lengths = pad_zeros([x])
    x = torch.from_numpy(x).float().to(device)
    img = img.unsqueeze(0).to(device)
    seq_lengths = torch.tensor(seq_lengths).to(device)
    pairs = torch.tensor(pairs).to(device)

    with torch.no_grad():
        out = model(x, img, seq_lengths, pairs, None)
        pred = out["pred_final"].cpu().numpy()[0]

    return pred  # shape [25], continuous probabilities

# Collect predictions
results = []
for item in paired_data:
    ehr_key = item["csv"]
    dicom_id = item.get("dicom_id", None)
    pred = predict(ehr_key, dicom_id)
    binarized = (pred > 0.5).astype(int)
    results.append({
        "ehr_id": ehr_key,
        "dicom_id": dicom_id if dicom_id else "None",
        **{cls: int(binarized[i]) for i, cls in enumerate(CLASSES)}
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("predictions_val.csv", index=False)
print(" Saved predictions to predictions_val.csv")

