
from django.db import models

# PyTorch model for inference
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class AlzheimerResNet50(nn.Module):
	def __init__(self, num_classes=4):
		super().__init__()
		self.resnet = models.resnet50(weights=None)
		in_features = self.resnet.fc.in_features
		self.resnet.fc = nn.Sequential(
			nn.Linear(in_features, 512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(512, num_classes)
		)

	def forward(self, x):
		return self.resnet(x)

# Create your models here.
