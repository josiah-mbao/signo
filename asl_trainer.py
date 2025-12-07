import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import numpy as np

class ASLClassifier:
    def __init__(self, dataset_path='asl_dataset.csv'):
        self.dataset_path = dataset_path
        self.model = None

    def load_data(self):
        """Load and preprocess landmark dataset"""
        df = pd.read_csv(self.dataset_path)
        X = df.iloc[:, 1:].values  # landmarks
        y = df['label'].values   # labels

        return X, y

    def train_model(self, model_type='svm', show_plots=False):
        """Train classification model with comprehensive metrics"""
        X, y = self.load_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Choose model
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale',
                           random_state=42, probability=True)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100,
                                              random_state=42)
        elif model_type == 'lr':
            self.model = LogisticRegression(max_iter=1000, random_state=42)

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train,
                                  cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                   random_state=42),
                                  scoring='accuracy')
        print(f"\nCross-validation Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Train
        print(f"\nTraining {model_type.upper()} model...")
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nTest Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        if show_plots:
            self._plot_confusion_matrix(cm, y_test)

        # Per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        classes = np.unique(y_test)
        print("\nPer-Class Accuracy:")
        for cls, acc in zip(classes, class_accuracy):
            print(f"  {cls}: {acc:.3f}")

        # Save evaluation metrics for Streamlit display
        self._save_evaluation_metrics(accuracy, cv_scores, class_accuracy, classes)

        return self.model

    def _save_evaluation_metrics(self, accuracy, cv_scores, class_accuracy, classes):
        """Save evaluation metrics to JSON for Streamlit display"""
        import json

        eval_data = {
            'accuracy': float(accuracy),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'per_class_accuracy': {cls: float(acc) for cls, acc in zip(classes, class_accuracy)}
        }

        with open('model_evaluation.json', 'w') as f:
            json.dump(eval_data, f, indent=2)

    def _plot_confusion_matrix(self, cm, y_true):
        """Plot confusion matrix"""
        classes = np.unique(y_true)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_model(self, filename='asl_classifier.pkl'):
        """Save trained model"""
        if self.model:
            joblib.dump(self.model, filename)
            print(f"Model saved to {filename}")
        else:
            print("No trained model to save")

    def load_model(self, filename='asl_classifier.pkl'):
        """Load saved model"""
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")

    def predict(self, landmarks):
        """Predict ASL letter from landmarks"""
        if not self.model:
            raise ValueError("Model not loaded. Train or load a model first.")

        if isinstance(landmarks, list):
            landmarks = np.array(landmarks).reshape(1, -1)

        return self.model.predict(landmarks)[0]

if __name__ == "__main__":
    classifier = ASLClassifier()

    print("Training ASL classifier...")
    classifier.train_model('svm')  # Alternatives: 'rf', 'lr'

    print("Saving model...")
    classifier.save_model()

    print("\nExample usage:")
    # Test with first training sample
    X, y = classifier.load_data()
    pred = classifier.predict(X[0])
    print(f"Predicted: {pred}, Actual: {y[0]}")
