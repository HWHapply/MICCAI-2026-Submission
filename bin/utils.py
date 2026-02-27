import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, precision_score, f1_score
from sklearn.preprocessing import label_binarize


def generate_project_name(mode, projection_dim, disentangle_losses=None,
                         decorr_weight=None, ortho_weight=None, ortho_concentration=None,
                         uniform_weight=None, uniform_temperature=None, warmup_epochs=None,
                         discretization_method=None, weight_learning_mode=None):
    """
    Generate project name based on configuration.
    Format: medmnist_disentangled_ph{projection_dim}_{loss_name}_{hyperparameters}_{discretization}_{weight_mode}
    """
    if mode != 'disentangled':
        return f"medmnist_{mode}"

    project_name = f"medmnist_disentangled_ph{projection_dim}"

    if disentangle_losses and len(disentangle_losses) > 0:
        loss_str = "-".join(sorted(disentangle_losses))
        project_name += f"_{loss_str}"

        params = []
        if 'decorr' in disentangle_losses and decorr_weight is not None:
            params.append(f"dw{decorr_weight}")

        if 'orthogonal' in disentangle_losses:
            if ortho_weight is not None:
                params.append(f"ow{ortho_weight}")
            if ortho_concentration is not None:
                params.append(f"oc{ortho_concentration}")

        if 'uniformity' in disentangle_losses:
            if uniform_weight is not None:
                params.append(f"uw{uniform_weight}")
            if uniform_temperature is not None:
                params.append(f"ut{uniform_temperature}")

        if warmup_epochs is not None and warmup_epochs > 0:
            params.append(f"wu{warmup_epochs}")

        if params:
            project_name += "_" + "_".join(params)

    # Add discretization method (for DARTS)
    if discretization_method is not None and discretization_method != 'topk':
        project_name += f"_{discretization_method}"

    # Add weight learning mode (for weighted discretization)
    if weight_learning_mode is not None and weight_learning_mode != 'fixed':
        # Abbreviate for brevity
        mode_abbrev = {
            'learnable_uniform': 'lu',
            'learnable_darts': 'ld',
            'none': 'noweight'
        }.get(weight_learning_mode, weight_learning_mode)
        project_name += f"_{mode_abbrev}"

    return project_name


def plot_roc_curve(y_true, y_prob, save_path, num_classes, average='macro'):
    """Plot and save ROC curve."""
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curve', fontweight='bold')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return roc_auc
    else:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        fpr, tpr, roc_auc = {}, {}, {}
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        roc_auc["macro"] = roc_auc_score(y_true_bin, y_prob, average='macro')
        
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        
        plt.figure(figsize=(10, 8))
        
        if average == 'micro':
            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-avg (AUC = {roc_auc["micro"]:.4f})',
                    color='red', linestyle=':', linewidth=4)
        else:
            plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-avg (AUC = {roc_auc["macro"]:.4f})',
                    color='red', linestyle=':', linewidth=4)
        
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.8,
                    label=f'Class {i} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
        plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
        plt.title(f'Multi-class ROC ({average.capitalize()}-avg AUC = {roc_auc[average]:.4f})', 
                 fontweight='bold', fontsize=14)
        plt.legend(loc="lower right", fontsize=9 if num_classes <= 10 else 7, 
                  ncol=1 if num_classes <= 10 else 2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return roc_auc[average]


def plot_confusion_matrix(y_true, y_pred, save_path, num_classes, class_names=None, average='macro'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        df_cm = pd.DataFrame(cm, columns=['Negative', 'Positive'], index=['Negative', 'Positive'])
        figsize = (6, 6)
    else:
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        if average == 'micro':
            tp_total = np.sum(np.diag(cm))
            fn_total = np.sum(cm) - tp_total
            sensitivity = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0

            tn_total, fp_total = 0, 0
            for i in range(num_classes):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                tn_total += tn
                fp_total += fp
            specificity = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0
        else:
            sensitivity_per_class = np.diag(cm) / cm.sum(axis=1)
            sensitivity = np.nanmean(sensitivity_per_class)

            specificity_per_class = []
            for i in range(num_classes):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                if (tn + fp) > 0:
                    specificity_per_class.append(tn / (tn + fp))
            specificity = np.mean(specificity_per_class) if specificity_per_class else 0.0
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(num_classes)]
        df_cm = pd.DataFrame(cm, columns=class_names, index=class_names)
        figsize = (max(8, num_classes), max(8, num_classes))
    
    plt.figure(figsize=figsize)
    sns.heatmap(df_cm, annot=True, fmt='d', cbar=True, cmap='Blues', 
                annot_kws={'fontsize': 10 if num_classes <= 5 else 8})
    plt.title(f'Confusion Matrix (ACC={acc:.4f})', fontweight='bold', fontsize=12)
    plt.xlabel('Prediction', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    
    if num_classes > 5:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return acc, sensitivity, specificity, precision, f1, cm


def to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj