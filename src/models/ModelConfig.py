from mrcnn.config import Config
from src.utils.all_paths import Paths

paths = Paths('../../')


class ModelConfig(Config):
    def __init__(self, name: str, num_classes: int, image_min_dim: int, image_max_dim: int, steps: int,
                 epochs: int = 20, detection_min_confidence: float = 0.9):
        super().__init__()
        self.NAME = name
        self.NUM_CLASSES = 1 + num_classes
        self.STEPS_PER_EPOCH = steps
        self.IMAGE_MIN_DIM = image_min_dim
        self.IMAGE_MAX_DIM = image_max_dim
        self.DETECTION_MIN_CONFIDENCE = detection_min_confidence
        self.EPOCHS = epochs
        # Batch size = GPU_COUNT * IMAGES_PER_GPU
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.BATCH_SIZE = 1
