import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np

from model import ModelBuilder
from config import Config


class EmotionPredictor:    
    def __init__(
        self,
        checkpoint_path: str = None,
        config: Config = None,
        device: str = None
    ):

        self.config = config if config else Config
        self.device = device if device else self.config.DEVICE
        
        print("\n" + "="*70)
        print("EMOTION PREDICTOR INITIALIZATION".center(70))
        print("="*70 + "\n")
        
        print("📦 Loading model...")
        self.model = ModelBuilder(
            num_classes=self.config.NUM_CLASSES,
            model_name=self.config.MODEL_NAME,
            pretrained=False,
            freeze_backbone=False,
            dropout_p=self.config.DROPOUT_P,
            print_summary=False
        )
        
        checkpoint_path = checkpoint_path if checkpoint_path else self.config.BEST_MODEL_PATH
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📁 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = self._get_transform()
        
        print(f"✓ Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Image Size: {self.config.IMAGE_SIZE}x{self.config.IMAGE_SIZE}")
        print(f"Classes: {self.config.NUM_CLASSES}")
        print("="*70 + "\n")
    
    def _get_transform(self):

        transform_steps = [
            transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
        
        if self.config.CHANNELS == 3:
            transform_steps.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        else:
            transform_steps.append(
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5]
                )
            )
        
        return transforms.Compose(transform_steps)
    
    def preprocess_image(self, img_path: str):
        img = Image.open(img_path)
        
        if self.config.CHANNELS == 3:
            img = img.convert("RGB")
        else:
            img = img.convert("L")
            img = Image.merge("RGB", (img, img, img))
        
        img_tensor = self.transform(img)
        
        return img_tensor.unsqueeze(0)
    
    def predict_image(self, img_path: str, return_probabilities: bool = False):  
        img_tensor = self.preprocess_image(img_path).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probabilities, dim=1)
            
            predicted_label = self.config.EMOTION_LABELS[pred_idx.item()]
            confidence_value = confidence.item()
        
        if return_probabilities:
            all_probs = {
                self.config.EMOTION_LABELS[i]: probabilities[0, i].item()
                for i in range(self.config.NUM_CLASSES)
            }
            return predicted_label, confidence_value, all_probs
        
        return predicted_label, confidence_value
    
    def predict_folder(
        self,
        folder_path: str,
        save_results: bool = True,
        output_dir: str = None
    ):
        output_dir = output_dir if output_dir else self.config.RESULTS_DIR
        
        print("\n" + "="*70)
        print("BATCH PREDICTION".center(70))
        print("="*70 + "\n")
        
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_files:
            print("⚠ No images found in the folder!")
            return []
        
        print(f"Found {len(image_files)} images to process\n")
        
        results = []
        for file in tqdm(image_files, desc="Processing", unit="image"):
            img_path = os.path.join(folder_path, file)
            
            try:
                label, conf, probs = self.predict_image(img_path, return_probabilities=True)
                
                result = {
                    "filename": file,
                    "prediction": label,
                    "confidence": conf
                }
                
                for emotion, prob in probs.items():
                    result[f"prob_{emotion}"] = prob
                
                results.append(result)
                
            except Exception as e:
                print(f"⚠ Error processing {file}: {e}")
                continue
        
        if save_results and results:
            self._save_results(results, output_dir)
        
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: list, output_dir: str):
        df = pd.DataFrame(results)
        
        csv_path = os.path.join(output_dir, "predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to:")
        print(f"   CSV: {csv_path}")
        
        try:
            excel_path = os.path.join(output_dir, "predictions.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"   Excel: {excel_path}")
        except Exception as e:
            print(f"   ⚠ Excel export failed: {e}")
        
        json_path = os.path.join(output_dir, "predictions.json")
        df.to_json(json_path, orient="records", indent=2)
        print(f"   JSON: {json_path}")
    
    def _print_summary(self, results: list):
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("PREDICTION SUMMARY".center(70))
        print("="*70 + "\n")
        
        emotion_counts = df['prediction'].value_counts()
        
        print("Emotion Distribution:")
        print("-" * 70)
        for emotion, count in emotion_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {emotion:<12}: {count:>4} ({percentage:>5.1f}%)")
        
        print("-" * 70)
        print(f"  Total Images: {len(results)}")
        
        avg_confidence = df['confidence'].mean()
        print(f"\n  Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        print("="*70 + "\n")
    
    def visualize_prediction(
        self,
        img_path: str,
        save_path: str = None,
        show: bool = True
    ):
        label, conf, probs = self.predict_image(img_path, return_probabilities=True)
        
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(
            f"Prediction: {label}\nConfidence: {conf*100:.2f}%",
            fontsize=14,
            fontweight='bold'
        )
        
        emotions = list(probs.keys())
        probabilities = list(probs.values())
        colors = ['#FF6B6B' if e == label else '#4ECDC4' for e in emotions]
        
        bars = ax2.barh(emotions, probabilities, color=colors)
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title('Emotion Probabilities', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        
        for bar in bars:
            width = bar.get_width()
            ax2.text(
                width, bar.get_y() + bar.get_height()/2,
                f'{width*100:.1f}%',
                ha='left', va='center', fontsize=10
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Emotion Recognition Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single image
  python predict.py --image path/to/image.jpg
  
  # Predict single image with visualization
  python predict.py --image path/to/image.jpg --show
  
  # Predict all images in folder
  python predict.py --folder path/to/images/
  
  # Use specific checkpoint
  python predict.py --image test.jpg --checkpoint checkpoints/best_model.pth
        """
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder with images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (uses best_model.pth if not specified)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display prediction visualization (only for single image)"
    )
    parser.add_argument(
        "--save-viz",
        type=str,
        default=None,
        help="Save visualization to file (only for single image)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: results/)"
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.error("Please provide either --image or --folder argument")
    
    try:
        predictor = EmotionPredictor(
            checkpoint_path=args.checkpoint,
            config=Config
        )
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        exit(1)
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            exit(1)
        
        print(f"\n📸 Processing: {args.image}\n")
        
        label, conf, probs = predictor.predict_image(
            args.image,
            return_probabilities=True
        )
        
        print(f"🎯 Prediction: {label}")
        print(f"📊 Confidence: {conf*100:.2f}%")
        print(f"\nAll Probabilities:")
        print("-" * 40)
        for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion:<12}: {prob*100:>5.2f}%")
        print("-" * 40)
        
        if args.show or args.save_viz:
            predictor.visualize_prediction(
                args.image,
                save_path=args.save_viz,
                show=args.show
            )
    
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            exit(1)
        
        results = predictor.predict_folder(
            args.folder,
            save_results=True,
            output_dir=args.output_dir
        )
        
        print("\nDetailed Results:")
        print("="*70)
        for r in results[:10]:
            print(f"{r['filename']:<30} → {r['prediction']:<10} ({r['confidence']*100:.2f}%)")
        
        if len(results) > 10:
            print(f"... and {len(results)-10} more images")
        print("="*70)
    
    print("\n✅ Prediction completed!")