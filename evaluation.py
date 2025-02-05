import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from data_handling import create_data_generators
from config import CONFUSION_MATRIX_SAVE_PATH, ROC_CURVE_SAVE_PATH, PRECISION_RECALL_CURVE_SAVE_PATH

def evaluate_model(model):
    print("Evaluating model...")
    _, _, test_gen = create_data_generators()
    y_pred = model.predict(test_gen)
    y_pred = np.round(y_pred).astype(int)
    target_names = ["Fire", "NonFire"]

    cm = confusion_matrix(test_gen.classes, y_pred)
    print("***** Confusion Matrix *****")
    print(cm)
    print("***** Classification Report *****")
    print(classification_report(test_gen.classes, y_pred, target_names=target_names))

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.6)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.savefig(CONFUSION_MATRIX_SAVE_PATH)
    plt.show()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_SAVE_PATH}")

    y_test = test_gen.classes
    predictions = model.predict(test_gen)

    # ROC Curve and AUC for both classes
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Fire (class 1)
    fpr["Fire"], tpr["Fire"], _ = roc_curve(y_test, predictions)
    roc_auc["Fire"] = auc(fpr["Fire"], tpr["Fire"])

    # Non-Fire (class 0)
    fpr["Non-Fire"], tpr["Non-Fire"], _ = roc_curve(1 - y_test, 1 - predictions)
    roc_auc["Non-Fire"] = auc(fpr["Non-Fire"], tpr["Non-Fire"])

    # Precision-Recall Curve and Average Precision for both classes
    precision = {}
    recall = {}
    average_precision = {}

    # Fire (class 1)
    precision["Fire"], recall["Fire"], _ = precision_recall_curve(y_test, predictions)
    average_precision["Fire"] = average_precision_score(y_test, predictions)

    # Non-Fire (class 0)
    precision["Non-Fire"], recall["Non-Fire"], _ = precision_recall_curve(1 - y_test, 1 - predictions)
    average_precision["Non-Fire"] = average_precision_score(1 - y_test, 1 - predictions)

    # Plot ROC Curves
    plt.figure(figsize=(12, 8))
    plt.plot(fpr["Fire"], tpr["Fire"], color='orange', lw=2, label=f'Fire (AUC = {roc_auc["Fire"]:.2f})')
    plt.plot(fpr["Non-Fire"], tpr["Non-Fire"], color='blue', lw=2, label=f'Non-Fire (AUC = {roc_auc["Non-Fire"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontname="Times New Roman")
    plt.ylabel('True Positive Rate', fontsize=14, fontname="Times New Roman")
    plt.title('Fire and Non-Fire ROC Curve', fontsize=16, fontname="Times New Roman")
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True)
    plt.savefig(ROC_CURVE_SAVE_PATH)
    plt.show()
    print(f"ROC curve saved to {ROC_CURVE_SAVE_PATH}")

    # Plot Precision-Recall Curves
    plt.figure(figsize=(12, 8))
    plt.plot(recall["Fire"], precision["Fire"], color='orange', lw=2, label=f'Fire (AP = {average_precision["Fire"]:.2f})')
    plt.plot(recall["Non-Fire"], precision["Non-Fire"], color='blue', lw=2, label=f'Non-Fire (AP = {average_precision["Non-Fire"]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14, fontname="Times New Roman")
    plt.ylabel('Precision', fontsize=14, fontname="Times New Roman")
    plt.title('Fire and Non-Fire Precision-Recall Curve', fontsize=16, fontname="Times New Roman")
    plt.legend(loc="lower left", fontsize=14)
    plt.grid(True)
    plt.savefig(PRECISION_RECALL_CURVE_SAVE_PATH)
    plt.show()
    print(f"Precision-recall curve saved to {PRECISION_RECALL_CURVE_SAVE_PATH}")
    print("Evaluation completed.")
