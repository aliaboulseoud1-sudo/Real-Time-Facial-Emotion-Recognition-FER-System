import sys
import argparse
import os

from config import Config
from preprocessingg import FERDatasetLoader
from model import ModelBuilder
from trainn import Trainer
from evaluate import EmotionEvaluator
from predict import EmotionPredictor
from live_dedection import LiveEmotionDetector


def print_header(title: str):
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def setup_data_pipeline():
    print("[STEP 1/4] Setting up Data Pipeline...")
    print("-" * 80)
    
    try:
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
        
        train_loader, val_loader, test_loader, total_samples = dataset_loader.get_loaders()
        lens = dataset_loader.get_len()
        
        print(f"✓ Data Pipeline Ready!")
        print(f"  Train: {lens['train_samples']:,} samples | Val: {lens['val_samples']:,} samples | Test: {lens['test_samples']:,} samples")
        print("-" * 80 + "\n")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def setup_model():
    print("[STEP 2/4] Building Model...")
    print("-" * 80)
    
    try:
        model = ModelBuilder(
            num_classes=Config.NUM_CLASSES,
            model_name=Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            freeze_backbone=Config.FREEZE_BACKBONE,
            dropout_p=Config.DROPOUT_P,
            print_summary=True
        )
        
        print("✓ Model Ready!")
        print("-" * 80 + "\n")
        
        return model
        
    except Exception as e:
        print(f"✗ Error building model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def train_model(model, train_loader, val_loader):
    print("[STEP 3/4] Starting Training...")
    print("-" * 80 + "\n")
    
    try:
        trainer = Trainer(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            config=Config
        )
        
        best_model_path = trainer.train()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED!".center(80))
        print("="*80)
        print(f"\n🎉 Training finished!")
        print(f"📁 Best model saved at: {best_model_path}")
        print(f"📊 Training logs saved in: {Config.LOGS_DIR}")
        print(f"💾 Checkpoints saved in: {Config.CHECKPOINT_DIR}")
        print("\n" + "="*80 + "\n")
        
        return best_model_path
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate_model(model, dataloader, dataset_name="test", checkpoint_path=None):
    print(f"[STEP 4/4] Starting Evaluation on {dataset_name.upper()} set...")
    print("-" * 80 + "\n")
    
    try:
        evaluator = EmotionEvaluator(
            model=model,
            dataloader=dataloader,
            config=Config,
            dataset_name=dataset_name
        )
        
        metrics = evaluator.evaluate(checkpoint_path=checkpoint_path)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETED!".center(80))
        print("="*80)
        print(f"\n📊 Results saved in: {Config.RESULTS_DIR}")
        print(f"📈 Plots saved in: {Config.PLOTS_DIR}")
        print(f"\n🎯 Test Accuracy: {metrics['overall']['accuracy']*100:.2f}%")
        print(f"📊 F1-Score (Macro): {metrics['overall']['f1_macro']:.4f}")
        print("\n" + "="*80 + "\n")
        
        return metrics
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def predict_image(image_path: str = None, folder_path: str = None, checkpoint_path: str = None):
    print_header("EMOTION PREDICTION")
    
    try:
        predictor = EmotionPredictor(
            checkpoint_path=checkpoint_path if checkpoint_path else Config.BEST_MODEL_PATH,
            config=Config
        )
        
        if image_path:
            print(f"📸 Processing image: {image_path}\n")
            
            label, conf, probs = predictor.predict_image(
                image_path,
                return_probabilities=True
            )
            
            print(f"🎯 Prediction: {label}")
            print(f"📊 Confidence: {conf*100:.2f}%")
            print(f"\nAll Probabilities:")
            print("-" * 40)
            for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"  {emotion:<12}: {prob*100:>5.2f}% {bar}")
            print("-" * 40)
            
            print("\n📊 Generating visualization...")
            viz_path = os.path.join(Config.RESULTS_DIR, "prediction_viz.png")
            predictor.visualize_prediction(image_path, save_path=viz_path, show=False)
            print(f"✓ Visualization saved: {viz_path}")
            
        elif folder_path:
            print(f"📁 Processing folder: {folder_path}\n")
            
            results = predictor.predict_folder(
                folder_path,
                save_results=True,
                output_dir=Config.PREDICTIONS_DIR
            )
            
            print(f"\n✓ Processed {len(results)} images")
            print(f"📊 Results saved in: {Config.PREDICTIONS_DIR}")
            
        else:
            print("❌ Please provide either --image or --folder")
            return
        
        print("\n" + "="*80)
        print("PREDICTION COMPLETED!".center(80))
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def live_detection(camera_id: int = 0, face_detection: bool = False, record: str = None, checkpoint_path: str = None):
    print_header("LIVE EMOTION DETECTION")
    
    try:
        detector = LiveEmotionDetector(
            checkpoint_path=checkpoint_path if checkpoint_path else Config.BEST_MODEL_PATH,
            config=Config,
            camera_id=camera_id,
            enable_face_detection=face_detection,
            smoothing_window=Config.SMOOTHING_WINDOW
        )
        
        detector.run(record_output=record)
        
    except Exception as e:
        print(f"\n✗ Error during live detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Emotion Recognition - Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train only
  python run.py --mode train
  
  # Evaluate only
  python run.py --mode evaluate --dataset test
  
  # Train then evaluate
  python run.py --mode both --dataset test
  
  # Predict single image
  python run.py --mode predict --image test.jpg
  
  # Predict folder
  python run.py --mode predict --folder test_images/
  
  # Live detection
  python run.py --mode live
  
  # Live detection with face detection
  python run.py --mode live --face-detection
  
  # Complete workflow (train + evaluate + predict)
  python run.py --mode all --image test.jpg
        """
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train",
        choices=["train", "evaluate", "predict", "live", "both", "all"],
        help="Execution mode (default: train)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset for evaluation (default: test)"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for prediction"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Path to folder with images for prediction"
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--face-detection",
        action="store_true",
        help="Enable face detection for live mode"
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to save video recording in live mode"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: best_model.pth)"
    )
    
    args = parser.parse_args()
    
    print_header(f"EMOTION RECOGNITION - {args.mode.upper()} MODE")
    
    Config.print_config()
    
    if args.mode == "train":
        train_loader, val_loader, test_loader = setup_data_pipeline()
        model = setup_model()
        train_model(model, train_loader, val_loader)
    
    elif args.mode == "evaluate":
        train_loader, val_loader, test_loader = setup_data_pipeline()
        model = setup_model()
        
        eval_loader = test_loader if args.dataset == "test" else val_loader
        checkpoint = args.checkpoint if args.checkpoint else Config.BEST_MODEL_PATH
        
        evaluate_model(model, eval_loader, args.dataset, checkpoint)
    
    elif args.mode == "predict":
        if not args.image and not args.folder:
            print("❌ Error: --mode predict requires either --image or --folder")
            sys.exit(1)
        
        predict_image(
            image_path=args.image,
            folder_path=args.folder,
            checkpoint_path=args.checkpoint
        )
    
    elif args.mode == "live":
        live_detection(
            camera_id=args.camera,
            face_detection=args.face_detection,
            record=args.record,
            checkpoint_path=args.checkpoint
        )
    
    elif args.mode == "both":
        train_loader, val_loader, test_loader = setup_data_pipeline()
        model = setup_model()
        
        best_model_path = train_model(model, train_loader, val_loader)
        
        print("\n" + "="*80)
        print("Starting Evaluation Phase...".center(80))
        print("="*80 + "\n")
        
        eval_loader = test_loader if args.dataset == "test" else val_loader
        evaluate_model(model, eval_loader, args.dataset, best_model_path)
    
    elif args.mode == "all":
        train_loader, val_loader, test_loader = setup_data_pipeline()
        model = setup_model()
        best_model_path = train_model(model, train_loader, val_loader)
        
        print("\n" + "="*80)
        print("Starting Evaluation Phase...".center(80))
        print("="*80 + "\n")
        
        eval_loader = test_loader if args.dataset == "test" else val_loader
        evaluate_model(model, eval_loader, args.dataset, best_model_path)
        
        if args.image or args.folder:
            print("\n" + "="*80)
            print("Starting Prediction Phase...".center(80))
            print("="*80 + "\n")
            
            predict_image(
                image_path=args.image,
                folder_path=args.folder,
                checkpoint_path=best_model_path
            )
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()