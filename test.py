import mlflow
print("printing tracking uri schema below")
print(mlflow.get_tracking_uri())
print("\n")

mlflow.set_tracking_uri("http://localhost:5000")
print("printing new tracking uri schema below")
print(mlflow.get_tracking_uri())
print("\n")