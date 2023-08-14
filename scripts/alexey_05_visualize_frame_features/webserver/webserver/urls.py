"""
URL configuration for webserver project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

import re

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    # path('get_video_data/<str:video_id>/', views.get_video_data, name='get_video_data'),
    # path('get_video_frame/<str:video_id>/<int:frame_idx>/', views.get_video_frame, name='get_video_frame'),
    # path('get_video_frame_data/<str:video_id>/<int:frame_idx>/', views.get_video_frame_data, name='get_video_frame_data'),
    # path('get_video_virtual_availability_data/<str:video_id>/', views.get_video_virtual_availability_data, name='get_video_virtual_availability_data'),
    # path('get_detailed_video_virtual_availability_data/<str:video_id>/', views.get_detailed_video_virtual_availability_data, name='get_detailed_video_virtual_availability_data'),
    # path('create_comment/<str:video_id>/<int:frame_idx>/', views.create_comment, name='create_comment'),
    # path('update_comment/<str:video_id>/<int:frame_idx>/<int:comment_id>/', views.update_comment, name='update_comment'),
    # path('remove_comment/<str:video_id>/<int:frame_idx>/<int:comment_id>/', views.remove_comment, name='remove_comment'),
    # path('admin/', admin.site.urls),
]
