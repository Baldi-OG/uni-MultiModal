import io
import json
import os

import open_clip
import torch
from PIL import Image

from homework_3.functions import parse_img, api_response_to_tensor


class ClipModel(torch.nn.Module):
    def __init__(self, head1_weights=None, head2_weights=None, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_backbone, _, self.preprocess = open_clip.create_model_and_transforms(model_name=model_name,
                                                                                       pretrained=pretrained)
        for p in self.clip_backbone.parameters():
            p.requires_grad = False
        self.clip_backbone.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.concept_labels = ["class1", "class2", "class3", "class4"]

        self.concept1_tokens = self.tokenizer(self.concept_labels[:2])
        self.concept2_tokens = self.tokenizer(self.concept_labels[2:])

        self.concept1_class_features = self.clip_backbone.encode_text(self.concept1_tokens)
        self.concept1_class_features /= self.concept1_class_features.norm(dim=-1, keepdim=True)
        self.concept1_class_features = self.concept1_class_features.to(self.device)

        self.concept2_class_features = self.clip_backbone.encode_text(self.concept2_tokens)
        self.concept2_class_features /= self.concept2_class_features.norm(dim=-1, keepdim=True)
        self.concept2_class_features = self.concept2_class_features.to(self.device)

        self.head_concept1 = torch.nn.Linear(self.clip_backbone.visual.output_dim,
                                             self.clip_backbone.visual.output_dim)
        self.head_concept2 = torch.nn.Linear(self.clip_backbone.visual.output_dim,
                                             self.clip_backbone.visual.output_dim)

        self.clip_backbone.to(self.device)
        self.head_concept1.to(self.device)
        self.head_concept2.to(self.device)

        self._load_heads(head1_weights, head2_weights)

    def _load_heads(self, head1_weights, head2_weights):
        if head1_weights:
            try:
                self.head_concept1.load_state_dict(
                    torch.load(head1_weights, map_location=self.device))
            except FileNotFoundError:
                exit("Could not find the provided filepath for loading head weights.\n"
                     f"Make sure {head1_weights} exists.")
        if head2_weights:
            try:
                self.head_concept2.load_state_dict(
                    torch.load(head2_weights, map_location=self.device))
            except FileNotFoundError:
                exit("Could not find the provided filepath for loading head weights.\n"
                     f"Make sure {head2_weights} exists.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.clip_backbone.encode_image(images)

        concept1_features = self.head_concept1(image_features)
        concept2_features = self.head_concept2(image_features)

        concept1_features = concept1_features / concept1_features.norm(dim=-1, keepdim=True)
        concept2_features = concept2_features / concept2_features.norm(dim=-1, keepdim=True)

        concept1_similarity = (100.0 * concept1_features @ self.concept1_class_features.T).softmax(dim=-1)
        concept2_similarity = (100.0 * concept2_features @ self.concept2_class_features.T).softmax(dim=-1)
        return torch.cat((concept1_similarity, concept2_similarity), dim=1)


class ApiResponsesDataset(torch.utils.data.Dataset):
    def __init__(self, clip_preprocess, img_dir="../img", img_series=None, responses_cache="../cache/responses.json"):
        super().__init__()

        self.clip_preprocess = clip_preprocess
        self.responses_json = json.load(open(responses_cache, "r"))

        self.raw_images = []
        if img_dir:
            self.raw_images = [Image.open(f"{img_dir}/{p}") for p in os.listdir(img_dir)]
        if img_series is not None:
            self.raw_images.extend([Image.open(io.BytesIO(i['bytes'])) for i in img_series])

        self.data = []
        self.labels = []

        self._preprocess()

    def _preprocess(self):
        self.data = [self.clip_preprocess(raw_img) for raw_img in self.raw_images]

        try:
            for raw_img in self.raw_images:
                _, img_hash = parse_img(raw_img)
                response = self.responses_json[img_hash]
                label = api_response_to_tensor(response)
                self.labels.append(label)
        except KeyError:
            exit(f"Dataset integrity error. Not all images have responses.")

        assert len(self.data) == len(self.labels), "Unequal amount of data and labels loaded"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
