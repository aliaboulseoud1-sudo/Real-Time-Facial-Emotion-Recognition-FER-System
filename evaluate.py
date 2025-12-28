import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_fscore_support
)
from torch.utils.data import DataLoader


class EmotionEvaluator:
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        config,
        dataset_name: str = "test"
    ):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.dataset_name = dataset_name
        self.device = config.DEVICE
        
        self.model = self.model.to(self.device)
        
        self.emotion_labels = config.EMOTION_LABELS
        self.num_classes = config.NUM_CLASSES
        
        print(f"\n{'='*70}")
        print(f"EVALUATOR INITIALIZED - {dataset_name.upper()} SET".center(70))
        print(f"{'='*70}\n")
    
    def evaluate(self, checkpoint_path: str = None):
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📁 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print("✓ Checkpoint loaded successfully!\n")
        else:
            print("⚠ No checkpoint provided, using current model weights\n")
        
        print("🔄 Running evaluation...")
        y_true, y_pred, y_prob = self._get_predictions()
        
        print("\n📊 Calculating metrics...")
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        
        self._print_metrics(metrics)
        
        self._save_metrics(metrics)
        
        print("\n📈 Generating visualizations...")
        self._plot_confusion_matrix(y_true, y_pred)
        self._plot_roc_curves(y_true, y_prob)
        
        self._save_classification_report(y_true, y_pred)
        
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETED!".center(70))
        print(f"{'='*70}\n")
        
        return metrics
    
    def _get_predictions(self):
        self.model.eval()
        
        y_true = []
        y_pred = []
        y_prob = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.dataloader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probabilities.cpu().numpy())
        
        return np.array(y_true), np.array(y_pred), np.array(y_prob)
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        accuracy = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted)
            },
            'per_class': {}
        }
        

        for idx in range(self.num_classes):
            emotion_name = self.emotion_labels[idx]
            metrics['per_class'][emotion_name] = {
                'precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'f1_score': float(f1[idx]),
                'support': int(support[idx])
            }
        
        return metrics
    
    def _print_metrics(self, metrics):
        print(f"\n{'='*70}")
        print(f"EVALUATION METRICS - {self.dataset_name.upper()} SET".center(70))
        print(f"{'='*70}\n")
        
        print("Overall Metrics:")
        print("-" * 70)
        print(f"  Accuracy         : {metrics['overall']['accuracy']:.4f}")
        print(f"  Precision (Macro): {metrics['overall']['precision_macro']:.4f}")
        print(f"  Recall (Macro)   : {metrics['overall']['recall_macro']:.4f}")
        print(f"  F1-Score (Macro) : {metrics['overall']['f1_macro']:.4f}")
        print()
        
        print("Per-Class Metrics:")
        print("-" * 70)
        print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        for emotion, values in metrics['per_class'].items():
            print(f"{emotion:<12} {values['precision']:<12.4f} {values['recall']:<12.4f} "
                  f"{values['f1_score']:<12.4f} {values['support']:<10}")
        
        print("-" * 70)
    
    def _save_metrics(self, metrics):
        output_path = os.path.join(
            self.config.RESULTS_DIR,
            f"{self.dataset_name}_metrics.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n✓ Metrics saved to: {output_path}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        emotion_names = [self.emotion_labels[i] for i in range(self.num_classes)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=emotion_names,
            yticklabels=emotion_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {self.dataset_name.upper()} Set', fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(
            self.config.PLOTS_DIR,
            f"{self.dataset_name}_confusion_matrix.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to: {output_path}")
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=emotion_names,
            yticklabels=emotion_names,
            cbar_kws={'label': 'Proportion'}
        )
        plt.title(f'Normalized Confusion Matrix - {self.dataset_name.upper()} Set', 
                 fontsize=16, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        output_path_norm = os.path.join(
            self.config.PLOTS_DIR,
            f"{self.dataset_name}_confusion_matrix_normalized.png"
        )
        plt.savefig(output_path_norm, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Normalized confusion matrix saved to: {output_path_norm}")
    
    def _plot_roc_curves(self, y_true, y_prob):
        plt.figure(figsize=(12, 10))
        
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                label=f'{self.emotion_labels[i]} (AUC = {roc_auc:.2f})',
                linewidth=2
            )
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.50)')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {self.dataset_name.upper()} Set', fontsize=16, pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(
            self.config.PLOTS_DIR,
            f"{self.dataset_name}_roc_curves.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to: {output_path}")
    
    def _save_classification_report(self, y_true, y_pred):
        emotion_names = [self.emotion_labels[i] for i in range(self.num_classes)]
        report = classification_report(
            y_true, y_pred,
            target_names=emotion_names,
            output_dict=True,
            zero_division=0
        )
        
        output_path = os.path.join(
            self.config.RESULTS_DIR,
            f"{self.dataset_name}_classification_report.csv"
        )
        
        with open(output_path, 'w') as f:
            f.write("Emotion,Precision,Recall,F1-Score,Support\n")
            
            for emotion in emotion_names:
                if emotion in report:
                    f.write(f"{emotion},{report[emotion]['precision']:.4f},"
                           f"{report[emotion]['recall']:.4f},"
                           f"{report[emotion]['f1-score']:.4f},"
                           f"{int(report[emotion]['support'])}\n")
            
            f.write("\n")
            f.write(f"Accuracy,{report['accuracy']:.4f},,,\n")
            f.write(f"Macro Avg,{report['macro avg']['precision']:.4f},"
                   f"{report['macro avg']['recall']:.4f},"
                   f"{report['macro avg']['f1-score']:.4f},"
                   f"{int(report['macro avg']['support'])}\n")
            f.write(f"Weighted Avg,{report['weighted avg']['precision']:.4f},"
                   f"{report['weighted avg']['recall']:.4f},"
                   f"{report['weighted avg']['f1-score']:.4f},"
                   f"{int(report['weighted avg']['support'])}\n")
        
        print(f"✓ Classification report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    from config import Config
    from preprocessingg import FERDatasetLoader
    from model import ModelBuilder
    
    parser = argparse.ArgumentParser(description="Evaluate Emotion Recognition Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset to evaluate: val or test"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("STANDALONE EVALUATION".center(70))
    print("="*70 + "\n")
    
    print("Loading dataset...")
    dataset_loader = FERDatasetLoader(
        train_dir=Config.TRAIN_DIR,
        test_dir=Config.TEST_DIR,
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMAGE_SIZE,
        channels=Config.CHANNELS,
        seed=Config.SEED,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    train_loader, val_loader, test_loader, _ = dataset_loader.get_loaders()
    
    eval_loader = test_loader if args.dataset == "test" else val_loader
    
    print("Loading model...")
    model = ModelBuilder(
        num_classes=Config.NUM_CLASSES,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        freeze_backbone=Config.FREEZE_BACKBONE,
        dropout_p=Config.DROPOUT_P,
        print_summary=False
    )
    
    evaluator = EmotionEvaluator(
        model=model,
        dataloader=eval_loader,
        config=Config,
        dataset_name=args.dataset
    )
    
    checkpoint_path = args.checkpoint if args.checkpoint else Config.BEST_MODEL_PATH
    evaluator.evaluate(checkpoint_path=checkpoint_path)
    
    print("\n✓ Evaluation completed!")