import os
import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import List
import torch

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(f"Project root: {root}")

# Imports that require root directory setup
from src.utils import logging_utils
from src.datamodules.dogbreed import DogBreedImageDataModule
from src.models.timm_classifier import DogBreedClassifier

log = logging_utils.logger

@logging_utils.task_wrapper
def train(cfg: DictConfig):
    # Determine the number of workers based on CPU/GPU usage
    num_workers = os.cpu_count() // 2 if cfg.trainer.accelerator == "gpu" else 0
    log.info(f"Using {num_workers} workers for data loading.")

    # Set up data module
    log.info("Instantiating datamodule")
    datamodule: DogBreedImageDataModule = hydra.utils.instantiate(cfg.data)
    
    # Set up model
    log.info("Instantiating model")
    model: DogBreedClassifier = hydra.utils.instantiate(cfg.model)

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
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=cfg.get("fast_dev_run", False)
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    
    # Print out the training metrics
    log.info("Training metrics:")
    for metric_name, metric_value in trainer.callback_metrics.items():
        log.info(f"{metric_name}: {metric_value}")

    # Test the model
    if cfg.get("test_after_training", False):
        test(cfg, trainer, model, datamodule)

    # Make sure everything closed properly
    log.info("Finished training!")

def test(cfg: DictConfig, trainer: L.Trainer, model: DogBreedClassifier, datamodule: DogBreedImageDataModule):
    log.info("Starting testing!")
    test_results = trainer.test(model=model, datamodule=datamodule)
    
    log.info("Test metrics:")
    for metric_name, metric_value in test_results[0].items():
        log.info(f"{metric_name}: {metric_value}")

    # Save test results
    results_file = os.path.join(cfg.paths.output_dir, "test_results.pt")
    torch.save(test_results[0], results_file)
    log.info(f"Test results saved to {results_file}")

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    # Setup logger for the entire script
    logging_utils.setup_logger(log_file=f"{cfg.paths.output_dir}/train.log")
    
    # Train the model
    train(cfg)

    # Perform testing if specified in the config
    if cfg.get("test", False):
        # Instantiate necessary components for testing
        datamodule: DogBreedImageDataModule = hydra.utils.instantiate(cfg.data)
        model: DogBreedClassifier = hydra.utils.instantiate(cfg.model)
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

        # Perform testing
        test(cfg, trainer, model, datamodule)


if __name__ == "__main__":
    main()
