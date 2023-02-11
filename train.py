import warnings
import argparse

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, f1_score, recall_score, precision_score

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators")
parser.add_argument("--max_depth")
args = parser.parse_args()


n_estimators = int(args.n_estimators)
max_depth = int(args.max_depth)

data = load_iris()
df = pd.DataFrame(data.data)

label = "class"
features = data.feature_names

df.columns = features
df[label] = data.target

X_train, X_test, y_train, y_test = train_test_split(
    df[features].to_numpy(), df[label].to_numpy())

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')

    print(f"Test accuracy: {accuracy}")
    print(f"Test precision: {precision}")
    print(f"Test recall: {recall}")
    print(f"Test f1: {f1}")

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(model, "model")
