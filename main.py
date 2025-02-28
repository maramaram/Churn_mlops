import numpy as np
import pandas as pd
import joblib
import argparse
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MLflow configuration
mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI
mlflow.set_experiment("Churn Prediction with AdaBoost")

def prepare_data(data_path='merged_churn.csv'):
    """Prepare the data for training."""
    logging.info("Preparing data...")
    data = pd.read_csv(data_path)
    
    # Drop the 'State' column
    data = data.drop('State', axis=1)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['International plan'] = label_encoder.fit_transform(data['International plan'])
    data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])
    data['Churn'] = label_encoder.fit_transform(data['Churn'])

    # Split features and target
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logging.info("Data preparation complete.")
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train the model using GridSearchCV with AdaBoost."""
    logging.info("Training model with AdaBoost...")
    
    # Define the base estimator (default is DecisionTreeClassifier with max_depth=1)
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)
    
    # Define the AdaBoost model
    ada_model = AdaBoostClassifier(estimator=base_estimator, random_state=42)
    
    # Define hyperparameters for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],  # Number of weak learners
        'learning_rate': [0.01, 0.1, 1.0],  # Learning rate
    }

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=ada_model, param_grid=param_grid, cv=cv, scoring='accuracy')
    
    # Start MLflow run
    with mlflow.start_run():
        # Perform grid search
        grid_search.fit(X_train, y_train)

        # Log hyperparameters
        mlflow.log_params(grid_search.best_params_)

        # Log metrics
        best_score = grid_search.best_score_
        mlflow.log_metric("best_accuracy", best_score)

        # Log the model
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

        logging.info(f"Best parameters found: {grid_search.best_params_}")
        logging.info(f"Best accuracy score: {best_score}")

    return grid_search

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    logging.info("Evaluating model...")
    y_pred = model.predict(X_test)

    # Print classification report and accuracy
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {accuracy}")

    return accuracy

def save_model(model, filename="model.pkl"):
    """Save the trained model to a file."""
    logging.info(f"Saving model to {filename}...")
    joblib.dump(model, filename)
    logging.info("Model saved.")

def save_scaler(scaler, filename="scaler.pkl"):
    """Save the scaler to a file."""
    logging.info(f"Saving scaler to {filename}...")
    joblib.dump(scaler, filename)
    logging.info("Scaler saved.")

def load_model(filename="model.pkl"):
    """Load a trained model from a file."""
    logging.info(f"Loading model from {filename}...")
    model = joblib.load(filename)
    logging.info("Model loaded.")
    return model

def load_scaler(filename="scaler.pkl"):
    """Load a scaler from a file."""
    logging.info(f"Loading scaler from {filename}...")
    scaler = joblib.load(filename)
    logging.info("Scaler loaded.")
    return scaler

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="MLOps Pipeline for Churn Prediction with AdaBoost")
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--data_path", type=str, default="merged_churn.csv", help="Path to the dataset")
    args = parser.parse_args()

    if args.prepare:
        X_train, X_test, y_train, y_test, scaler = prepare_data(args.data_path)
        logging.info(f"Training set size: {X_train.shape[0]} samples")
        logging.info(f"Test set size: {X_test.shape[0]} samples")
        save_scaler(scaler)

    if args.train:
        X_train, X_test, y_train, y_test, _ = prepare_data(args.data_path)
        model = train_model(X_train, y_train)
        save_model(model)

    if args.evaluate:
        X_train, X_test, y_train, y_test, _ = prepare_data(args.data_path)
        model = load_model()
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
