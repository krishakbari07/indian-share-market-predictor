from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_home, name='dashboard_home'),
    path('api/predict/', views.predict_stock, name='predict_stock'),
]
