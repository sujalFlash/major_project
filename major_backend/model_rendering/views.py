
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv


from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from PIL import Image
import torch
from torchvision import transforms
import csv
from .models import AlzheimerResNet50

class ModelInferenceView(APIView):
	def post(self, request):
		image_file = request.FILES.get('image')
		if not image_file:
			return Response({'error': 'No image uploaded.'}, status=status.HTTP_400_BAD_REQUEST)

		try:
			image = Image.open(image_file).convert('RGB')
		except Exception as e:
			return Response({'error': f'Invalid image file: {e}'}, status=status.HTTP_400_BAD_REQUEST)

		preprocess = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		])
		input_tensor = preprocess(image).unsqueeze(0)

		model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'best_model (2).pth')
		if not os.path.exists(model_path):
			return Response({'error': 'Model file not found.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

		try:
			checkpoint = torch.load(model_path, map_location='cpu')
			num_classes = len(checkpoint.get('class_names', ['mild', 'moderate', 'no', 'very mild']))
			model = AlzheimerResNet50(num_classes=num_classes)
			state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
			model.resnet.load_state_dict(state_dict)
			model.eval()
		except Exception as e:
			return Response({'error': f'Model loading failed: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

		with torch.no_grad():
			outputs = model(input_tensor)
			probs = torch.softmax(outputs, dim=1)
			pred_class = torch.argmax(probs, dim=1).item()
			confidence = probs[0, pred_class].item()

		class_names = checkpoint.get('class_names', ['mild', 'moderate', 'no', 'very mild'])
		pred_label = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)

		result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'result')
		os.makedirs(result_dir, exist_ok=True)
		csv_path = os.path.join(result_dir, 'results.csv')
		with open(csv_path, 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow([image_file.name, pred_label, confidence])

		return Response({'filename': image_file.name, 'prediction': pred_label, 'confidence': confidence}, status=status.HTTP_200_OK)
