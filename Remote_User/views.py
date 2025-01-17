from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import os
from social_media_popularity_prediction import settings

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,detect_popularity_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})

def update_profile(request):
    if request.method == 'POST':
        # Update the user profile here
        user = request.user
        user.username = request.POST.get('username')
        user.email = request.POST.get('email')
        user.phoneno = request.POST.get('phoneno')
        user.gender = request.POST.get('gender')
        user.address = request.POST.get('address')
        user.country = request.POST.get('country')
        user.state = request.POST.get('state')
        user.city = request.POST.get('city')
        user.save()  # Save the updated details in the database

        return redirect('profile')  # Redirect to the profile page or appropriate URL

    return render(request, 'userprofile.html')


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


from textblob import TextBlob
from django.shortcuts import render, redirect
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import random
import os
from django.conf import settings
from Remote_User.models import ClientRegister_Model

# Initialize model tracking with CCMB bandit parameters
model_rewards = {'naive_bayes': [], 'svm': [], 'logistic': [], 'decision_tree': [], 'neural_network': []}
model_confidences = {'naive_bayes': 1.0, 'svm': 1.0, 'logistic': 1.0, 'decision_tree': 1.0, 'neural_network': 1.0}

def select_model_with_ccmb():
    """Select the best model based on cumulative confidence multi-armed bandit approach."""
    scores = {}
    for model in model_rewards:
        if model_rewards[model]:
            mean_reward = np.mean(model_rewards[model])
            confidence = model_confidences[model]
            scores[model] = mean_reward + confidence * np.sqrt(1 / (len(model_rewards[model]) + 1))
        else:
            scores[model] = random.uniform(0, 1)  # Prioritize exploration for new models
    return max(scores, key=scores.get)

def control_diffusion(sensitivity):
    """Control diffusion based on content sensitivity."""
    if sensitivity == 1:  # Sensitive content
        return random.uniform(0.1, 0.3)  # Low diffusion probability
    else:  # Non-sensitive content
        return random.uniform(0.7, 0.9)  # Higher diffusion probability

def update_model_rewards(model, reward):
    """Update the rewards for the selected model."""
    model_rewards[model].append(reward)
    model_confidences[model] *= 0.95  # Decay confidence slightly

def Predict_Social_Media_Popularity(request):
    train_df = pd.read_csv('Datasets1.csv')
    train_df['results'] = train_df['score'].apply(lambda x: 1 if x > 100 else 0)

    # Add sentiment analysis to the training data
    train_df['sentiment'] = train_df['post_desc'].apply(lambda x: TextBlob(x).sentiment.polarity)

    cv = CountVectorizer()
    X_train = cv.fit_transform(train_df['post_desc'])
    y_train = train_df['results']

    models = {
        'naive_bayes': MultinomialNB(),
        'svm': svm.LinearSVC(),
        'logistic': LogisticRegression(random_state=0, solver='lbfgs'),
        'decision_tree': DecisionTreeClassifier(),
        'neural_network': MLPClassifier()
    }

    if request.method == "POST":
        if 'file' in request.FILES:  # Bulk prediction
            file = request.FILES['file']
            df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

            # Validate required columns
            required_columns = ['post_desc', 'score']
            if not all(col in df.columns for col in required_columns):
                return render(request, 'RUser/Predict_Social_Media_Popularity.html', 
                              {'error': 'Uploaded file must contain "post_desc" and "score" columns.'})

            # Apply response logic and sentiment analysis
            df['results'] = df['score'].apply(lambda x: 1 if x > 100 else 0)
            df['sentiment'] = df['post_desc'].apply(lambda x: TextBlob(x).sentiment.polarity)
            X_test = cv.transform(df['post_desc'])
            y_test = df['results']

            selected_model_name = select_model_with_ccmb()  # Select best model using bandit
            selected_model = models[selected_model_name]
            selected_model.fit(X_train, y_train)
            predictions = selected_model.predict(X_test)

            # Update the model's reward and confidence
            reward = accuracy_score(y_test, predictions)
            update_model_rewards(selected_model_name, reward)

            df['Predicted Popularity'] = ['Sensitive' if pred == 1 else 'Non-sensitive' for pred in predictions]
            df['Diffusion_Probability'] = df['results'].apply(control_diffusion)

            # Sentiment-based adjustment
            df['Adjusted Sensitivity'] = df.apply(lambda row: 'Highly Sensitive' if row['sentiment'] < -0.5 and row['results'] == 1 else row['Predicted Popularity'], axis=1)

            # Save the output
            output_file = os.path.join(settings.MEDIA_ROOT, 'Bulk_Predictions.csv')
            df.to_csv(output_file, index=False)
            file_url = f"{settings.MEDIA_URL}Bulk_Predictions.csv"

            return render(request, 'RUser/Predict_Social_Media_Popularity.html', 
                          {'objs': 'Bulk predictions complete.', 'file_url': file_url})

        else:  # Single prediction
            post_desc = request.POST.get('post_desc')
            score = request.POST.get('score')

            if not post_desc or not score:
                return render(request, 'RUser/Predict_Social_Media_Popularity.html', 
                              {'error': 'Please provide both "post_desc" and "score".'})

            score = int(score)
            sentiment = TextBlob(post_desc).sentiment.polarity
            sensitivity = 1 if score > 100 else 0
            context_vector = cv.transform([post_desc])

            selected_model_name = select_model_with_ccmb()
            selected_model = models[selected_model_name]
            selected_model.fit(X_train, y_train)

            prediction = selected_model.predict(context_vector)[0]
            diffusion_probability = control_diffusion(sensitivity)
            prediction_label = 'Sensitive' if prediction == 1 else 'Non-sensitive'

            # Sentiment-based adjustment for single prediction
            if sentiment < -0.5 and prediction == 1:
                prediction_label = 'Highly Sensitive'

            return render(request, 'RUser/Predict_Social_Media_Popularity.html', 
                          {'objs': f'Prediction: {prediction_label}, Sentiment: {sentiment:.2f}, Diffusion Probability: {diffusion_probability:.2f}'})

    return render(request, 'RUser/Predict_Social_Media_Popularity.html')
