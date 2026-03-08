import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from transformers import AutoModel, AutoTokenizer, pipeline
from PIL import Image
import pickle
import re

# =============================
# LOAD LABEL ENCODER
# =============================
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

NUM_CLASSES = len(label_encoder.classes_)

# =============================
# DEVICE
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# TOKENIZER
# =============================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# =============================
# IMAGE TRANSFORM (MUST MATCH TRAINING)
# =============================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class MultiModalModel(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalModel, self).__init__()

        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()

        self.fc1 = nn.Linear(768 + 2048, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask, image):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_features = bert_outputs.last_hidden_state[:, 0, :]
        image_features = self.cnn(image)

        combined = torch.cat((text_features, image_features), dim=1)

        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# =============================
# LOAD MODEL WEIGHTS
# =============================
model = MultiModalModel(NUM_CLASSES)
model.load_state_dict(torch.load("multimodal_model_final.pt", map_location=device))
model.to(device)
model.eval()

# =============================
# LOAD NER MODEL
# =============================
ner_pipeline = pipeline(
    "ner",
    model="Davlan/xlm-roberta-base-ner-hrl",
    aggregation_strategy="simple"
)

# =============================
# TEXT PREPROCESSING FOR NER
# =============================
def preprocess_for_ner(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.title()
    return text.strip()

def extract_location(text):
    clean_text = preprocess_for_ner(text)
    entities = ner_pipeline(clean_text)

    locations = [
        ent["word"]
        for ent in entities
        if ent["entity_group"] == "LOC"
    ]

    if not locations:
        return "Unknown"

    return list(set(locations))


# =============================
# PREDICT FUNCTION
# =============================
def predict(text, image_path):

    # ---- Text Encoding ----
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # ---- Image Processing ----
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    # ---- Model Prediction ----
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    disaster_type = label_encoder.inverse_transform([pred.item()])[0]

    # ---- Location Extraction ----
    locations = extract_location(text)

    return {
        "disaster_type": disaster_type,
        "confidence": confidence.item(),
        "locations_detected": locations
    }


# =============================
# TEST
# =============================
# result = predict(
#     "water flood near mumbai coast",
#     ".\909377586052124674_1.jpg"
# )

# print(result)