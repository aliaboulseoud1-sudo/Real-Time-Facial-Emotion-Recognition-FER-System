import torch
import sys
import os

from preprocessingg import FERDatasetLoader
from model import ModelBuilder
from trainn import Trainer
from config import Config


class EmotionRecognitionPipeline: 
    def __init__(self):
        self.config = Config
        self.device = self.config.DEVICE
        
        print("\n" + "="*70)
        print("EMOTION RECOGNITION PIPELINE - INITIALIZATION".center(70))
        print("="*70 + "\n")
        
        self.config.print_config()
        
        self.dataset_loader = None
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def setup_data(self):
        print("[STEP 1/3] Setting up Data Pipeline...")
        print("-" * 70)
        
        try:
            self.dataset_loader = FERDatasetLoader(
                train_dir=self.config.TRAIN_DIR,
                test_dir=self.config.TEST_DIR,
                batch_size=self.config.BATCH_SIZE,
                image_size=self.config.IMAGE_SIZE,
                channels=self.config.CHANNELS,
                seed=self.config.SEED,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY
            )
            
            self.train_loader, self.val_loader, self.test_loader, total_samples = \
                self.dataset_loader.get_loaders()
            
            lens = self.dataset_loader.get_len()
            classes = self.dataset_loader.get_classes()
            
            print("\n✓ Data Pipeline Setup Complete!")
            print("\nDataset Statistics:")
            print(f"  Total Train Folder : {total_samples:,} samples")
            print(f"  Train Split        : {lens['train_samples']:,} samples ({lens['train_samples']/total_samples*100:.1f}%)")
            print(f"  Validation Split   : {lens['val_samples']:,} samples ({lens['val_samples']/total_samples*100:.1f}%)")
            print(f"  Test Set           : {lens['test_samples']:,} samples")
            print(f"\nBatch Information:")
            print(f"  Train Batches      : {lens['train_batches']}")
            print(f"  Val Batches        : {lens['val_batches']}")
            print(f"  Test Batches       : {lens['test_batches']}")
            print(f"\nClasses: {classes}")
            print("-" * 70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error setting up data pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_model(self):
        print("[STEP 2/3] Setting up Model...")
        print("-" * 70)
        
        try:
            self.model = ModelBuilder(
                num_classes=self.config.NUM_CLASSES,
                model_name=self.config.MODEL_NAME,
                pretrained=self.config.PRETRAINED,
                freeze_backbone=self.config.FREEZE_BACKBONE,
                dropout_p=self.config.DROPOUT_P,
                print_summary=True
            )
            
            self.model = self.model.to(self.device)
            
            print(f"✓ Model moved to device: {self.device}")
            print("-" * 70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error setting up model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_trainer(self):
        print("[STEP 3/3] Setting up Trainer...")
        print("-" * 70)
        
        try:
            self.trainer = Trainer(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                model=self.model,
                config=self.config
            )
            
            print("✓ Trainer setup complete!")
            print("-" * 70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error setting up trainer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_pipeline(self):
        print("[VERIFICATION] Testing pipeline with sample batch...")
        print("-" * 70)
        
        try:
            sample_batch, sample_labels = next(iter(self.train_loader))
            sample_batch = sample_batch.to(self.device)
            sample_labels = sample_labels.to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(sample_batch)
            
            print(f"✓ Forward pass successful!")
            print(f"\nBatch Shape:")
            print(f"  Input  : {sample_batch.shape}")
            print(f"  Output : {outputs.shape}")
            print(f"  Labels : {sample_labels.shape}")
            print(f"\nSample Predictions:")
            _, predicted = torch.max(outputs, 1)
            for i in range(min(5, len(predicted))):
                pred_emotion = self.config.EMOTION_LABELS[predicted[i].item()]
                true_emotion = self.config.EMOTION_LABELS[sample_labels[i].item()]
                print(f"  Sample {i+1}: Predicted={pred_emotion:10s} | True={true_emotion}")
            
            print("\n" + "="*70)
            print("PIPELINE READY FOR TRAINING!".center(70))
            print("="*70 + "\n")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Pipeline verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize(self):
        if not self.setup_data():
            return False
        
        if not self.setup_model():
            return False
        
        if not self.setup_trainer():
            return False
        
        if not self.verify_pipeline():
            return False
        
        return True
    
    def start_training(self):
        if self.trainer is None:
            print("✗ Trainer not initialized! Call initialize() first.")
            return None
        
        print("\n" + "="*70)
        print("STARTING TRAINING".center(70))
        print("="*70 + "\n")
        
        try:
            best_model_path = self.trainer.train()
            
            print("\n" + "="*70)
            print("TRAINING COMPLETED!".center(70))
            print("="*70 + "\n")
            
            return best_model_path
            
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def start_evaluation(self, dataset_type="test", checkpoint_path=None):
        from evaluate import EmotionEvaluator
        
        print("\n" + "="*70)
        print(f"🎯 Starting Model Evaluation - {dataset_type.upper()} Set".center(70))
        print("="*70)
        
        if dataset_type == "val":
            loader = self.val_loader
        elif dataset_type == "test":
            loader = self.test_loader
        else:
            raise ValueError("dataset_type must be 'val' or 'test'")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"📥 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'])
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print("✓ Checkpoint loaded successfully!")
        
        evaluator = EmotionEvaluator(
            model=self.model,
            dataloader=loader,
            config=self.config,
            dataset_name=dataset_type
        )
        
        metrics = evaluator.evaluate()
        
        print("\n✅ Evaluation Complete!")
        print(f"📊 Results saved in: {self.config.RESULTS_DIR}")
        
        return metrics
    
    def start_prediction(self, image_path=None, folder_path=None):
        from predict import EmotionPredictor
        
        print("\n" + "="*70)
        print("🎯 Starting Prediction".center(70))
        print("="*70)
        
        predictor = EmotionPredictor(
            checkpoint_path=self.config.BEST_MODEL_PATH,
            config=self.config
        )
        
        if image_path:
            label, conf, probs = predictor.predict_image(
                image_path,
                return_probabilities=True
            )
            
            print(f"\n🎯 Prediction: {label}")
            print(f"📊 Confidence: {conf*100:.2f}%")
            
            return {"prediction": label, "confidence": conf, "probabilities": probs}
        
        elif folder_path:
            results = predictor.predict_folder(folder_path)
            return results
        
        else:
            print("❌ Please provide either image_path or folder_path")
            return None
    
    def start_live_detection(self, camera_id=0, enable_face_detection=False):
        from live_dedection import LiveEmotionDetector
        
        print("\n" + "="*70)
        print("🎥 Starting Live Emotion Detection".center(70))
        print("="*70)
        
        detector = LiveEmotionDetector(
            checkpoint_path=self.config.BEST_MODEL_PATH,
            config=self.config,
            camera_id=camera_id,
            enable_face_detection=enable_face_detection
        )
        
        detector.run()
    
    def get_components(self):
        return {
            'model': self.model,
            'train_loader': self.train_loader,
            'val_loader': self.val_loader,
            'test_loader': self.test_loader,
            'device': self.device,
            'config': self.config,
            'trainer': self.trainer
        }


def main():
    pipeline = EmotionRecognitionPipeline()
    
    if pipeline.initialize():
        print("✓ Pipeline initialization successful!")
        print("\nNext steps:")
        print("  1. Call pipeline.start_training() to train the model")
        print("  2. Call pipeline.start_evaluation('test') for evaluation")
        print("  3. Call pipeline.start_prediction(image_path='...') for inference")
        print("  4. Call pipeline.start_live_detection() for real-time detection")
        
        print("\n" + "-"*70)
        print("Do you want to start training now? [Y/n]: ", end="")
        
        try:
            choice = input().strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\n✓ Pipeline ready. Exiting...")
            return pipeline
        
        if choice in ['y', 'yes', '']:
            best_model_path = pipeline.start_training()
            
            if best_model_path:
                print("\n" + "-"*70)
                print("Do you want to evaluate on test set now? [Y/n]: ", end="")
                
                try:
                    eval_choice = input().strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\n\n✓ Training complete. Exiting...")
                    return pipeline
                
                if eval_choice in ['y', 'yes', '']:
                    pipeline.start_evaluation('test', best_model_path)
        else:
            print("\n✓ Pipeline ready. Run 'python train.py' when ready to train.")
        
        return pipeline
    else:
        print("✗ Pipeline initialization failed!")
        sys.exit(1)


if __name__ == "__main__":
    pipeline = main()