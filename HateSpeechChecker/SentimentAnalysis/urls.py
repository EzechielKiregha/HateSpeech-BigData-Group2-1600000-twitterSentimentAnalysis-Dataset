from django.urls import path
from . import views

urlpatterns = [
    path('', views.predictor, name='predictor'),
    path('viewsPrevious/', views.viewsPrevious, name="viewsPrevious")
]
