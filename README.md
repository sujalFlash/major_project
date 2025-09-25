# Alzheimer's MRI Classification Django API

This project provides a Django REST API for Alzheimer's MRI image classification using a PyTorch ResNet50 model.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd major_project
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Django Migrations
```bash
cd major_backend
python manage.py migrate
```

### 5. Run the Development Server
```bash
python manage.py runserver
```

### 6. API Usage
- Endpoint: `POST /api/infer/`
- Upload an image file with the key `image`.
- The API will return the predicted class and confidence.
- Results are saved to `result/results.csv`.

#### Example using `curl`:
```bash
curl -X POST -F "image=@images/download.jpeg" http://127.0.0.1:8000/api/infer/
```

## Notes
- Ensure your trained model weights (`best_model (2).pth`) are in the `models/` directory.
- The API uses the model architecture defined in `model_rendering/models.py`.
- For any issues, check the Django server logs for error messages.

## Project Structure
- `major_backend/` - Django backend code
- `model_rendering/` - API views, model definition
- `models/` - Trained PyTorch model weights
- `images/` - Example images for testing
- `result/` - Inference results CSV

---

For further help, contact the project maintainer.
