import torch
import os


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_DIR = "fer2013"
    
    RAW_DATA_DIR = DATA_DIR
    FER2013_CSV = os.path.join(RAW_DATA_DIR, "fer2013.csv")
    
    PROCESSED_DATA_DIR = DATA_DIR
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINTS_DIR = CHECKPOINT_DIR  
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")
    
    RESULTS_DIR = "results"
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")  
    PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")  
    RECORDINGS_DIR = os.path.join(RESULTS_DIR, "recordings")  
    SCREENSHOTS_DIR = os.path.join(RESULTS_DIR, "screenshots")  
    
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 7
    PRETRAINED = True
    FREEZE_BACKBONE = False
    DROPOUT_P = 0.6
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 35
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    OPTIMIZER = "adam"
    MOMENTUM = 0.9 
    
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "plateau"
    STEP_SIZE = 10 
    GAMMA = 0.1
    PATIENCE = 3
    
    IMAGE_SIZE = 224
    CHANNELS = 3
    NUM_WORKERS = 4
    PIN_MEMORY = True
    SEED = 42
    
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP_PROB = 0.5
    ROTATION_DEGREES = 10
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    USE_AMP = True 
    
    EMOTION_LABELS = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral"
    }
    
    EMOTION_TO_IDX = {v: k for k, v in EMOTION_LABELS.items()}
    
    EARLY_STOPPING_PATIENCE = 10
    SAVE_BEST_ONLY = True
    VAL_SPLIT = 0.2
    
    CONFUSION_MATRIX_NORMALIZE = True  
    SAVE_PREDICTIONS = True
    PLOT_DPI = 300
    
    CAMERA_ID = 0
    ENABLE_FACE_DETECTION = False
    SMOOTHING_WINDOW = 5
    DISPLAY_FPS = True
    
    PREDICTION_BATCH_SIZE = 32
    SAVE_VISUALIZATIONS = True
    
    @staticmethod
    def create_dirs():
        directories = [
            Config.CHECKPOINT_DIR,
            Config.RESULTS_DIR,
            Config.PLOTS_DIR,
            Config.LOGS_DIR,
            Config.METRICS_DIR,
            Config.PREDICTIONS_DIR,
            Config.RECORDINGS_DIR,
            Config.SCREENSHOTS_DIR,
        ]
        
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✓ All directories created successfully!")
    
    @staticmethod
    def print_config():
        print("\n" + "="*70)
        print("PROJECT CONFIGURATION".center(70))
        print("="*70 + "\n")
        
        print("🖥️  Device & Performance:")
        print(f"   Device              : {Config.DEVICE}")
        print(f"   Mixed Precision     : {Config.USE_AMP}")
        print(f"   Num Workers         : {Config.NUM_WORKERS}")
        
        print(f"\n🧠 Model Settings:")
        print(f"   Model               : {Config.MODEL_NAME}")
        print(f"   Pretrained          : {Config.PRETRAINED}")
        print(f"   Freeze Backbone     : {Config.FREEZE_BACKBONE}")
        print(f"   Number of Classes   : {Config.NUM_CLASSES}")
        print(f"   Dropout             : {Config.DROPOUT_P}")
        
        print(f"\n🏋️  Training Settings:")
        print(f"   Batch Size          : {Config.BATCH_SIZE}")
        print(f"   Learning Rate       : {Config.LEARNING_RATE}")
        print(f"   Number of Epochs    : {Config.NUM_EPOCHS}")
        print(f"   Optimizer           : {Config.OPTIMIZER.upper()}")
        print(f"   Weight Decay        : {Config.WEIGHT_DECAY}")
        print(f"   Use Scheduler       : {Config.USE_SCHEDULER}")
        if Config.USE_SCHEDULER:
            print(f"   Scheduler Type      : {Config.SCHEDULER_TYPE.upper()}")
        print(f"   Early Stopping      : {Config.EARLY_STOPPING_PATIENCE} epochs")
        
        print(f"\n📊 Data Settings:")
        print(f"   Image Size          : {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
        print(f"   Channels            : {Config.CHANNELS}")
        print(f"   Val Split           : {Config.VAL_SPLIT*100:.0f}%")
        print(f"   Augmentation        : {Config.USE_AUGMENTATION}")
        print(f"   Random Seed         : {Config.SEED}")
        
        print(f"\n📁 Paths:")
        print(f"   Data Directory      : {Config.DATA_DIR}")
        print(f"   Checkpoints         : {Config.CHECKPOINT_DIR}")
        print(f"   Results             : {Config.RESULTS_DIR}")
        
        print("\n" + "="*70 + "\n")

    @staticmethod
    def validate_config():
        if not os.path.exists(Config.DATA_DIR):
            raise ValueError(f"Data directory not found: {Config.DATA_DIR}")
        
        if not os.path.exists(Config.TRAIN_DIR):
            print(f"⚠️  Warning: Train directory not found: {Config.TRAIN_DIR}")
        
        if not os.path.exists(Config.TEST_DIR):
            print(f"⚠️  Warning: Test directory not found: {Config.TEST_DIR}")
        
        if Config.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be positive")
        
        if Config.LEARNING_RATE <= 0:
            raise ValueError("LEARNING_RATE must be positive")
        
        if Config.NUM_EPOCHS <= 0:
            raise ValueError("NUM_EPOCHS must be positive")
        
        if Config.IMAGE_SIZE <= 0:
            raise ValueError("IMAGE_SIZE must be positive")
        
        if not 0 <= Config.VAL_SPLIT < 1:
            raise ValueError("VAL_SPLIT must be between 0 and 1")
        
        if Config.NUM_CLASSES != len(Config.EMOTION_LABELS):
            raise ValueError(
                f"NUM_CLASSES ({Config.NUM_CLASSES}) doesn't match "
                f"EMOTION_LABELS length ({len(Config.EMOTION_LABELS)})"
            )
        
        print("✓ Configuration validated successfully!")
    
    @staticmethod
    def get_emotion_name(idx: int) -> str:
        return Config.EMOTION_LABELS.get(idx, "Unknown")
    
    @staticmethod
    def get_emotion_idx(name: str) -> int:
        return Config.EMOTION_TO_IDX.get(name, -1)
    
    @staticmethod
    def save_config(filepath: str = None):
        import json
        
        if filepath is None:
            filepath = os.path.join(Config.RESULTS_DIR, "config.json")
        
        config_dict = {
            "model": {
                "name": Config.MODEL_NAME,
                "num_classes": Config.NUM_CLASSES,
                "pretrained": Config.PRETRAINED,
                "freeze_backbone": Config.FREEZE_BACKBONE,
                "dropout": Config.DROPOUT_P
            },
            "training": {
                "batch_size": Config.BATCH_SIZE,
                "num_epochs": Config.NUM_EPOCHS,
                "learning_rate": Config.LEARNING_RATE,
                "optimizer": Config.OPTIMIZER,
                "weight_decay": Config.WEIGHT_DECAY,
                "use_scheduler": Config.USE_SCHEDULER,
                "scheduler_type": Config.SCHEDULER_TYPE,
                "early_stopping_patience": Config.EARLY_STOPPING_PATIENCE
            },
            "data": {
                "image_size": Config.IMAGE_SIZE,
                "channels": Config.CHANNELS,
                "val_split": Config.VAL_SPLIT,
                "use_augmentation": Config.USE_AUGMENTATION,
                "seed": Config.SEED
            },
            "device": str(Config.DEVICE),
            "emotion_labels": Config.EMOTION_LABELS
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"✓ Configuration saved to: {filepath}")
    
    @staticmethod
    def load_config(filepath: str):
        import json
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        if "model" in config_dict:
            Config.MODEL_NAME = config_dict["model"].get("name", Config.MODEL_NAME)
            Config.NUM_CLASSES = config_dict["model"].get("num_classes", Config.NUM_CLASSES)
            Config.PRETRAINED = config_dict["model"].get("pretrained", Config.PRETRAINED)
            Config.FREEZE_BACKBONE = config_dict["model"].get("freeze_backbone", Config.FREEZE_BACKBONE)
            Config.DROPOUT_P = config_dict["model"].get("dropout", Config.DROPOUT_P)
        
        if "training" in config_dict:
            Config.BATCH_SIZE = config_dict["training"].get("batch_size", Config.BATCH_SIZE)
            Config.NUM_EPOCHS = config_dict["training"].get("num_epochs", Config.NUM_EPOCHS)
            Config.LEARNING_RATE = config_dict["training"].get("learning_rate", Config.LEARNING_RATE)
        
        print(f"✓ Configuration loaded from: {filepath}")


Config.create_dirs()
