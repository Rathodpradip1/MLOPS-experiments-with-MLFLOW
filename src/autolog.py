import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Rathodpradip1', repo_name='MLOPS-experiments-with-MLFLOW', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Rathodpradip1/MLOPS-experiments-with-MLFLOW.mlflow")

# load wine dataset
wine = load_wine()
x = wine.data
y = wine.target


# split dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.10 , random_state=42)

#define parameters for the random forest classifier
max_depth = 10
n_estimators = 5

# mention the experiment name
mlflow.autolog()  # Automatically log parameters, metrics, and models
mlflow.set_experiment("MLOPS-Exp4")

# write code for the mlflow experiment
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators , random_state=42) 
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test , y_pred)

    


    # creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6 ,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save the confusion matrix plot
    plt.savefig("confusion_matrix.png")

    # log artifact using mlflow
   
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": "Pradip rathod", "project": "wine-classification"})

    # log the model
    

    print(f"accuracy: {accuracy}")