'''Collection of helper functions for notebooks.'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import pickle
import os


def plot_cross_validation(search_results:GridSearchCV) -> None:
    '''Takes result object from scikit-learn's GridSearchCV(),
    draws plot of hyperparameter set validation score rank vs
    training and validation scores.'''

    results=pd.DataFrame(search_results.cv_results_)
    sorted_results=results.sort_values('rank_test_score')

    plt.title('Hyperparameter optimization')
    plt.xlabel('Hyperparameter set validation accuracy rank')
    plt.ylabel('Validation accuracy (%)')
    plt.gca().invert_xaxis()

    plt.fill_between(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score']*100 + sorted_results['std_test_score']*100,
        sorted_results['mean_test_score']*100 - sorted_results['std_test_score']*100,
        alpha=0.5
    )

    plt.plot(
        sorted_results['rank_test_score'],
        sorted_results['mean_test_score']*100,
        label='Validation'
    )

    plt.fill_between(
        sorted_results['rank_test_score'],
        sorted_results['mean_train_score']*100 + sorted_results['std_train_score']*100,
        sorted_results['mean_train_score']*100 - sorted_results['std_train_score']*100,
        alpha=0.5
    )

    plt.plot(
        sorted_results['rank_test_score'],
        sorted_results['mean_train_score']*100,
        label='Training'
    )

    plt.legend(loc='best', fontsize='small')
    plt.show()


# my own functions.py
def load_data(file_path):
    """Load dataset from a file with multiple format support."""
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.pkl':
        with open(file_path, 'rb') as input_file:
            dataset = pickle.load(input_file)
    elif file_extension == '.csv':
        dataset = pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        dataset = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    return dataset

def train_test_split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """Perform train-test split."""
    from sklearn.model_selection import train_test_split
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and generate evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic'])
    print("Classification Report:")
    print(report)

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="green", label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.show()
    
    return cm, report, roc_auc, fpr, tpr

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def tabulate_metrics(models, X_test, y_test):
    """Tabulate evaluation metrics for multiple models.

    Args:
        models (dict): Dictionary of model names and their corresponding trained model objects.
        X_test (array-like): Features for the test set.
        y_test (array-like): Ground truth labels for the test set.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each model.
    """
    metrics = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'ROC AUC': []
    }

    for model_name, model in models.items():
        # Predict labels and probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        metrics['Model'].append(model_name)
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['Precision'].append(precision_score(y_test, y_pred, pos_label=1))
        metrics['Recall'].append(recall_score(y_test, y_pred, pos_label=1))
        metrics['F1-Score'].append(f1_score(y_test, y_pred, pos_label=1))
        if y_prob is not None:  # ROC AUC requires probabilities
            metrics['ROC AUC'].append(roc_auc_score(y_test, y_prob))
        else:
            metrics['ROC AUC'].append("N/A")  # If the model doesn't support predict_proba

    return pd.DataFrame(metrics)
