import os
import hydra
from omegaconf import DictConfig
import lightning as L
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import glob
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

# Imports that require root directory setup
from src.utils import logging_utils
from src.models.timm_classifier import DogBreedClassifier

log = logging_utils.logger

@logging_utils.task_wrapper
def infer(cfg: DictConfig):
    # Set up trainer (needed to access the checkpoint callback)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    # Set up model
    log.info("Instantiating model")
    model: DogBreedClassifier = hydra.utils.instantiate(cfg.model)

    # Load the best model checkpoint
    if trainer.checkpoint_callback.best_model_path:
        log.info(f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    else:
        log.warning("No checkpoint found! Using initialized model weights.")

    # Print out the metrics from trainer.callback_metrics
    log.info("Model metrics:")
    for metric_name, metric_value in trainer.callback_metrics.items():
        log.info(f"{metric_name}: {metric_value}")

    model.eval()

    # Set up image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get input images
    input_dir = os.path.join(cfg.paths.model_artifacts, "input_images")
    output_dir = os.path.join(cfg.paths.model_artifacts, "predicted_images")
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))[:10]  # Get first 10 jpg images

    for img_path in image_files:
        # Load and preprocess the image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_label = model.idx_to_class[predicted_idx.item()]

        # Create a figure with the image and prediction
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_label}")
        plt.axis('off')

        # Save the figure
        output_path = os.path.join(output_dir, f"pred_{os.path.basename(img_path)}")
        plt.savefig(output_path)
        plt.close()

        log.info(f"Processed {img_path} -> {output_path}")

    log.info("Inference complete")

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig) -> None:
    # Setup logger for the entire script
    logging_utils.setup_logger(log_file=f"{cfg.paths.output_dir}/infer.log")
    
    # Run inference
    infer(cfg)

if __name__ == "__main__":
    main()
