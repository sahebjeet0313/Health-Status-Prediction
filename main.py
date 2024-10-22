# This file is for training and saving model files

# Import Dependencies
import yaml
from joblib import dump, load, Parallel, delayed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier

class DiseasePrediction:
    # Initialize and Load the Config File
    def __init__(self):
        # Load Config File
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        # Load Training Data
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        # Load Test Data
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        # Model Save Path
        self.model_save_path = self.config['model_save_path']

    # Function to Load Train Dataset
    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        # Check for data sanity
        assert (len(train_features.iloc[0]) == 132)
        assert (len(train_labels) == train_features.shape[0])

        return train_features, train_labels, df_train

    # Function to Load Test Dataset
    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        # Check for data sanity
        assert (len(test_features.iloc[0]) == 132)
        assert (len(test_labels) == test_features.shape[0])

        return test_features, test_labels, df_test

    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        return X_train, y_train, X_val, y_val

    # Model Selection
    def _get_model(self, model_name):
        if model_name == 'decision_tree':
            clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif model_name == 'random_forest':
            clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'], n_jobs=-1)
        return clf

    # Function to Train Models in Parallel and Select Best Model
    def train_and_select_best_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()

        # Parallel execution for training and validating models
        model_results = Parallel(n_jobs=-1)(
            delayed(self._train_and_evaluate)(X_train, y_train, X_val, y_val, model_name)
            for model_name in ['decision_tree', 'random_forest']
        )

        # Select the best model based on validation accuracy
        best_model = max(model_results, key=lambda x: x[2])  # x[2] is accuracy
        print(f'\nBest Model: {best_model[0]} with Accuracy: {best_model[2]}')
        return best_model[0]  # Return the best model name

    # Helper Function to Train and Evaluate a Model
    def _train_and_evaluate(self, X_train, y_train, X_val, y_val, model_name):
        # Select and train the model
        classifier = self._get_model(model_name)
        classifier.fit(X_train, y_train)
        
        # Evaluate on validation data
        confidence = classifier.score(X_val, y_val)
        y_pred = classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        clf_report = classification_report(y_val, y_pred)

        # Save Trained Model
        dump(classifier, str(self.model_save_path + model_name + ".joblib"))

        # Return the result for this model
        return model_name, confidence, accuracy, clf_report

    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model
            clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")

        if test_data is not None:
            result = clf.predict(test_data)
            return result
        else:
            result = clf.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, result)
        clf_report = classification_report(self.test_labels, result)
        return accuracy, clf_report
    
if __name__ == "__main__":
    # Instantiate the Class
    dp = DiseasePrediction()
    # Train and Automatically Select the Best Model
    best_model_name = dp.train_and_select_best_model()

    # Get Model Performance on Test Data
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=best_model_name)
    print("Model Test Accuracy: ", test_accuracy)
    print("Test Data Classification Report: \n", classification_report)
