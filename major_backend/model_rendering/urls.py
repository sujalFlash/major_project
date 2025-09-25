from django.urls import path
from .views import ModelInferenceView

urlpatterns = [
    path('infer/', ModelInferenceView.as_view(), name='model-infer'),
]
 