# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "torch>=2.14.1",
#   "torchaudio>=2.11.0",
#   "torchvision>=0.29.1",
#   "fastapi==0.162.0",
#   "uvicorn==0.34.0",
#   "pydantic>=2.14.2",
#   "open-clip-torch>=3.3.0",
#   "pandas>=3.1.0",
#   "pyarrow>=26.0.0",
#   "matplotlib>=3.11.2",
# ]
# ///

import torch
import numpy as np
import argparse
import open_clip
from PIL import Image
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
import io


class ClipModel(torch.nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        super().__init__()
        self.clip_backbone, _, self.preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
        for p in self.clip_backbone.parameters():
            p.requires_grad = False  # freeze the backbone
        self.clip_backbone.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
        self.emotion_classes = self.tokenizer(self.emotions)
        self.emotion_features = self.clip_backbone.encode_text(self.emotion_classes)
        if torch.cuda.is_available():
            self.emotion_features /= self.emotion_features.norm(dim=-1, keepdim=True)
            self.emotion_features = self.emotion_features.cuda()


        self.emotion_classifier_head = torch.nn.Linear(self.clip_backbone.visual.output_dim, self.clip_backbone.visual.output_dim)
        if torch.cuda.is_available():
            self.clip_backbone.cuda()
            self.emotion_classifier_head.cuda()


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # TODO: at least from mnist tests the clip preprocessing seems to be worse, test whether this is true in general

        with torch.no_grad():
            image_features = self.clip_backbone.encode_image(images)
        image_features = self.emotion_classifier_head(image_features)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        emotion_similarity = (100.0 * image_features @ self.emotion_features.T).softmax(dim=-1)

        return emotion_similarity

# the split implementation feels very scuffed, anyone should feel free to make it not awful
class DatasetMixer(torch.utils.data.Dataset):
    def __init__(self, clip_preprocessor):
        self.raw_datasets = []
        self.raw_datasets.append(pd.read_parquet("hf://datasets/Mat303/raf-db-tcc/data/train-00000-of-00001.parquet"))

        self.images = None   # list of all images from all datasets
        self.labels = None    # list of all labels from all datasets
        self.lengths = [len(dataset) for dataset in self.raw_datasets]
        self.cumulative_lengths = [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]   # might not need this anymore
        self.clip_preprocessor = clip_preprocessor

        self.preprocess()

    #TODO: add preprocessing for all datasets so they can be sampled the same way
    def preprocess(self):

        raf_images = [self.clip_preprocessor(self.raf_db_decode_row(r)) for _, r in self.raw_datasets[0].iterrows()]
        raf_labels = [r["label"] for _, r in self.raw_datasets[0].iterrows()]
        raf_emotions = torch.tensor(np.array(raf_labels))[:, 0].long() - 1 # only first row is for emotions starting at 1, offset and turn to long to use torch one hot encoding
        # emotion classes are in order: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
        # create one hot encoding in order of ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
        raf_one_hot_labels = torch.nn.functional.one_hot(raf_emotions, num_classes=7)
        raf_one_hot_labels = raf_one_hot_labels[:, [0, 1, 2, 3, 5, 6, 4]].float()   # Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral

        total_images = torch.stack(raf_images)  #TODO: concatenate all here
        total_labels = raf_one_hot_labels  #TODO: concatenate all here
        self.total_images = total_images
        self.total_labels = total_labels


    def split_dataset(self, train:bool, train_split:float, split_perm: torch.Tensor ):
        # split permutation for shuffling the dataset, should be a torch tensor of indices, necessary to ensure that the same split is used for train and validation sets, not make one dataset train only and since they might be sorted
        # do 70/30 split
        split_idx = int(self.total_images.size(0) * train_split)
        if train:
            self.images = self.total_images[split_perm[:split_idx]]
            self.labels = self.total_labels[split_perm[:split_idx]]
        else:
            self.images = self.total_images[split_perm[split_idx:]]
            self.labels = self.total_labels[split_perm[split_idx:]]

        del self.total_images
        del self.total_labels


    def raf_db_decode_row(self, row):
        img = row["image"]
        if isinstance(img, dict):
            raw = img.get("bytes")
        elif isinstance(img, bytes):
            raw = img
        else:
            raise TypeError(f"Unexpected image type: {type(img)}")
        if raw is None:
            raise ValueError("No image bytes found")
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return image

    def __len__(self):
        if self.images is None:
            return len(self.total_images)   # so we can get the length of the dataset before splitting it into train and validation sets
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def optim_early_stopping(loss_history, threshold=0.001, patience=8):
    if len(loss_history) <= patience:
        return False  # Not enough data to make a decision
    optimum = torch.min(loss_history[:-patience])  # Minimum loss observed before the recent window
    recent_minimum = torch.min(loss_history[-patience:])
    return (optimum - recent_minimum) < threshold

def train_model(model: ClipModel, train_data, val_data, num_epochs=10, learning_rate=1e-4):
    optimizer = torch.optim.Adam(model.emotion_classifier_head.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    val_loss_history = []
    acc_history = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.emotion_classifier_head.train()
        epoch_loss = 0.0
        for images, labels in train_data:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Train Loss: {epoch_loss/len(train_data)}')

        # Validation
        model.emotion_classifier_head.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_data:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 0)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Validation Loss: {val_loss/len(val_data)}, Accuracy: {100 * round(accuracy, 5)}%')

        val_loss_history.append(val_loss)
        acc_history.append(-accuracy)
        if optim_early_stopping(torch.tensor(val_loss_history), threshold=0.0001, patience=8) and optim_early_stopping(torch.tensor(acc_history), threshold=0.0001, patience=8):
            print(f"Early stopping at epoch {epoch+1} due to no significant improvement in validation loss.")
            break
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--random_seed", default=671611312, type=int, help="random seed")
    parser.add_argument("-w", "--weights_dir", default="./weights", type=str, help="directory to load model weights from")
    parser.add_argument("-e", "--epochs", default=50, type=int, help="number of training epochs")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float, help="learning rate for training")
    parser.add_argument("-b", "--batch_size", default=32, type=int, help="batch size for training")

    args = parser.parse_args()
    weights_dir = args.weights_dir

    torch.random.manual_seed(671611312)    # set fixed seed to ensure determinism requirement
    model = ClipModel()
    train_dataset = DatasetMixer(model.preprocess)
    perm = torch.randperm(len(train_dataset)) # classes are sorted so we need to shuffle them
    train_dataset.split_dataset(train=True, train_split=0.7, split_perm=perm)
    validation_dataset = DatasetMixer(model.preprocess)
    validation_dataset.split_dataset(train=False, train_split=0.7, split_perm=perm)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    train_model(model, train_data=train_dataloader, val_data=validation_dataloader, num_epochs=args.epochs, learning_rate=args.learning_rate)
    torch.save(model.emotion_classifier_head.state_dict(), f"{weights_dir}/emotion_classifier_head.pth")
