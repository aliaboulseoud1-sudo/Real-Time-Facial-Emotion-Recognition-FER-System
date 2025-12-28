import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import numpy as np
import json

from model import ModelBuilder
from preprocessingg import FERDatasetLoader
from config import Config


class Trainer:
    def __init__(
        self,
        train_loader,
        val_loader,
        model: ModelBuilder,
        config: Config = None,
        lr: float = None,
        epochs: int = None,
        checkpoint_dir: str = None,
        device: str = None,
        use_amp: bool = True,
        early_stopping_patience: int = None,
        seed: int = None
    ):

        self.config = config if config else Config
        
        self.lr = lr if lr is not None else self.config.LEARNING_RATE
        self.epochs = epochs if epochs is not None else self.config.NUM_EPOCHS
        self.checkpoint_dir = checkpoint_dir if checkpoint_dir else self.config.CHECKPOINT_DIR
        self.early_stopping_patience = early_stopping_patience if early_stopping_patience is not None else self.config.EARLY_STOPPING_PATIENCE
        self.seed = seed if seed is not None else self.config.SEED

        self._set_seed(self.seed)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.use_amp = use_amp

        self.device = device if device else self.config.DEVICE
        self.model.to(self.device)

        self.optimizer = self._get_optimizer()

        self.criterion = nn.CrossEntropyLoss()

        self.scheduler = self._get_scheduler() if self.config.USE_SCHEDULER else None

        self.enable_amp = self.use_amp and str(self.device) == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(self.config.LOGS_DIR, f"exp_{timestamp}")
        self.writer = SummaryWriter(log_dir=log_dir)

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.best_model_path = None
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

        print(f"\n{'='*70}")
        print("TRAINER INITIALIZED".center(70))
        print(f"{'='*70}")
        print(f"Device               : {self.device}")
        print(f"AMP Enabled          : {self.enable_amp}")
        print(f"Optimizer            : {self.config.OPTIMIZER.upper()}")
        print(f"Learning Rate        : {self.lr}")
        print(f"Epochs               : {self.epochs}")
        print(f"Early Stopping       : {self.early_stopping_patience} epochs")
        print(f"Scheduler            : {self.config.SCHEDULER_TYPE.upper() if self.config.USE_SCHEDULER else 'None'}")
        print(f"{'='*70}\n")

    def _set_seed(self, seed):
        """Set seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _get_optimizer(self):
        if self.config.OPTIMIZER.lower() == "adam":
            return Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.config.WEIGHT_DECAY
            )
        elif self.config.OPTIMIZER.lower() == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.config.MOMENTUM,
                weight_decay=self.config.WEIGHT_DECAY
            )
        else:
            raise ValueError(f"Optimizer {self.config.OPTIMIZER} not supported")

    def _get_scheduler(self):
        if self.config.SCHEDULER_TYPE.lower() == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.STEP_SIZE,
                gamma=self.config.GAMMA
            )
        elif self.config.SCHEDULER_TYPE.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.PATIENCE,
                factor=self.config.GAMMA,
                verbose=True
            )
        else:
            return None

    @staticmethod
    def compute_accuracy(outputs, labels):
        _, preds = torch.max(outputs, 1)
        return (preds == labels).sum().item() / labels.size(0)

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [TRAIN]")

        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            total_samples += batch_size

            self.optimizer.zero_grad()

            if self.enable_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if torch.isnan(loss):
                    print("⚠ NaN detected — skipping batch")
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                if torch.isnan(loss):
                    print("⚠ NaN detected — skipping batch")
                    continue

                loss.backward()
                self.optimizer.step()

            acc = self.compute_accuracy(outputs, labels)
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        avg_loss = running_loss / total_samples
        avg_acc = running_acc / total_samples
        return avg_loss, avg_acc

    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        loop = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [VAL]  ")

        with torch.no_grad():
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                batch_size = images.size(0)
                total_samples += batch_size

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                acc = self.compute_accuracy(outputs, labels)

                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size

                loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

        avg_loss = running_loss / total_samples
        avg_acc = running_acc / total_samples
        return avg_loss, avg_acc

    def save_checkpoint(self, epoch, train_loss, val_loss, train_acc, val_acc, is_best=False):
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "config": {
                "model_name": self.config.MODEL_NAME,
                "num_classes": self.config.NUM_CLASSES,
                "image_size": self.config.IMAGE_SIZE
            }
        }

        if is_best:
            path = self.config.BEST_MODEL_PATH
            self.best_model_path = path
            print(f"💾 Saving BEST model → {path}")
        else:
            path = self.config.LAST_MODEL_PATH

        torch.save(checkpoint, path)

    def save_training_history(self):
        history_path = os.path.join(self.config.RESULTS_DIR, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=4)
        print(f"📊 Training history saved → {history_path}")

    def train(self):
        print("\n" + "="*70)
        print("STARTING TRAINING".center(70))
        print("="*70 + "\n")

        patience_counter = 0

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            val_loss, val_acc = self.validate_one_epoch(epoch)

            current_lr = self.optimizer.param_groups[0]['lr']

            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_acc"].append(val_acc)
            self.training_history["lr"].append(current_lr)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)

            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{self.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc*100:.2f}%")
            print(f"  LR        : {current_lr:.6f}")

            self.save_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, is_best=False)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(epoch, train_loss, val_loss, train_acc, val_acc, is_best=True)
                print(f"  ⭐ New best model! Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            else:
                patience_counter += 1
                print(f"  ⏳ Patience: {patience_counter}/{self.early_stopping_patience}")

                if patience_counter >= self.early_stopping_patience:
                    print(f"\n⏹ Early stopping triggered after {epoch+1} epochs")
                    break

            print(f"{'='*70}\n")

        print("\n" + "="*70)
        print("TRAINING COMPLETED!".center(70))
        print("="*70)
        print(f"\n🏆 Best Model Performance:")
        print(f"   Val Loss : {self.best_val_loss:.4f}")
        print(f"   Val Acc  : {self.best_val_acc*100:.2f}%")
        print(f"   Saved at : {self.best_model_path}")
        print(f"\n{'='*70}\n")

        self.save_training_history()

        self.writer.close()

        return self.best_model_path


if __name__ == "__main__":
    from config import Config
    
    Config.print_config()
    
    print("Loading dataset...")
    data_loader = FERDatasetLoader(
        train_dir=Config.TRAIN_DIR,
        test_dir=Config.TEST_DIR,
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMAGE_SIZE,
        channels=Config.CHANNELS,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    train_loader, val_loader, test_loader, _ = data_loader.get_loaders()
    print("✓ Dataset loaded successfully!\n")

    print("Building model...")
    model = ModelBuilder(
        num_classes=Config.NUM_CLASSES,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        freeze_backbone=Config.FREEZE_BACKBONE,
        dropout_p=Config.DROPOUT_P,
        print_summary=True
    )
    print("✓ Model built successfully!\n")

    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        config=Config
    )

    best_model_path = trainer.train()
    
    print(f"\n✅ Training completed! Best model saved at: {best_model_path}")