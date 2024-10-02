import os
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List
import glob
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

# Imports that require root directory setup
from src.utils import logging_utils
from src.datamodules.dogbreed import DogBreedImageDataModule
from src.models.timm_classifier import TimmClassifier

log = logging_utils.logger

@logging_utils.task_wrapper
def evaluate(cfg: DictConfig):
    # Set up data module
    log.info("Instantiating datamodule")
    datamodule: DogBreedImageDataModule = hydra.utils.instantiate(cfg.data)
    
    # Prepare data and setup
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    
    # Ensure validation dataset is initialized correctly
    if datamodule.val_dataset is None:
        raise ValueError("Validation dataset is not initialized. Check the setup method.")

    # Get the number of classes from the data module
    num_classes = len(datamodule.val_dataset.dataset.classes)
    log.info(f"Number of classes detected: {num_classes}")

    # Find the latest checkpoint
    checkpoint_dir = cfg.paths.checkpoint_dir
    checkpoint_pattern = os.path.join(checkpoint_dir, "epoch_*-val_acc_*.ckpt")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    log.info(f"Loading latest checkpoint: {latest_checkpoint}")

    # Set up callbacks
    callbacks: List[L.Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Set up logger
    logger: List[Logger] = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Set up trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Set up model
    log.info("Instantiating model")
    model: TimmClassifier = hydra.utils.instantiate(cfg.model, num_classes=num_classes)

    # Load the best model checkpoint
    if trainer.checkpoint_callback.best_model_path:
        log.info(f"Loading best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes)
    else:
        log.warning("No checkpoint found! Using initialized model weights.")

    # Evaluate the model
    log.info("Starting evaluation")
    results = trainer.validate(model, datamodule=datamodule)

    # Print validation metrics
    log.info("Validation Metrics:")
    for k, v in results[0].items():
        log.info(f"{k}: {v}")

    # Print callback metrics
    log.info("Callback Metrics:")
    for k, v in trainer.callback_metrics.items():
        log.info(f"{k}: {v}")

    log.info("Evaluation complete")

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    # Setup logger for the entire script
    logging_utils.setup_logger(log_file=f"{cfg.paths.output_dir}/eval.log")
    
    # Evaluate the model
    evaluate(cfg)

if __name__ == "__main__":
    main()
