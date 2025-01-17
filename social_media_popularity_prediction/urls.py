"""social_media_popularity_prediction URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import path
from django.contrib import admin
from Remote_User import views as remoteuser
from social_media_popularity_prediction import settings
from Service_Provider import views as serviceprovider
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', remoteuser.index, name="index"),
    path('login/', remoteuser.login, name="login"),
    path('Register1/', remoteuser.Register1, name="Register1"),
    path('Predict_Social_Media_Popularity/', remoteuser.Predict_Social_Media_Popularity, name="Predict_Social_Media_Popularity"),
    path('ViewYourProfile/', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    path('serviceproviderlogin/', serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    path('View_Remote_Users/', serviceprovider.View_Remote_Users, name="View_Remote_Users"),
    path('charts/<str:chart_type>/', serviceprovider.charts, name="charts"),
    path('charts1/<str:chart_type>/', serviceprovider.charts1, name="charts1"),
    path('likeschart/<str:like_chart>/', serviceprovider.likeschart, name="likeschart"),
    path('View_Predicted_Social_Media_Popularity_Ratio/', serviceprovider.View_Predicted_Social_Media_Popularity_Ratio, name="View_Predicted_Social_Media_Popularity_Ratio"),
    path('train_model/', serviceprovider.train_model, name="train_model"),
    path('update_profile/', remoteuser.update_profile, name='update_profile'),
    path('View_Predicted_Social_Media_Popularity/', serviceprovider.View_Predicted_Social_Media_Popularity, name="View_Predicted_Social_Media_Popularity"),
    path('Download_Predicted_DataSets/', serviceprovider.Download_Predicted_DataSets, name="Download_Predicted_DataSets"),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
