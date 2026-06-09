from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np


class NewsClassifier:
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        self.model = self._initialize_model()
        self.is_trained = False
        
        # Category mapping
        self.category_mapping = {
            0: 'World',
            1: 'Sports',
            2: 'Business',
            3: 'Sci/Tech'
        }
        self.reverse_mapping = {v: k for k, v in self.category_mapping.items()}
    
    def _initialize_model(self):
        """Initialize the selected model"""
        if self.model_type == 'naive_bayes':
            return MultinomialNB(alpha=0.1)
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'svm':
            return LinearSVC(random_state=42, max_iter=10000)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """Train the classifier"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities (for models that support it)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models like SVM that don't have predict_proba
            decision = self.model.decision_function(X)
            # Convert to probabilities using softmax
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=list(self.category_mapping.values()))
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def get_category_name(self, prediction):
        """Convert numeric prediction to category name"""
        return self.category_mapping.get(prediction, 'Unknown')
    
    def get_category_id(self, category_name):
        """Convert category name to numeric ID"""
        return self.reverse_mapping.get(category_name, -1)
    
    def save(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        joblib.dump(self.model, filepath)
    
    def load(self, filepath):
        """Load a trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        return self
