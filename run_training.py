import logging
from typing import Optional
import os
import hydra
import torch
import pytorch_lightning as pl
from nuplan.planning.script.builders.folder_builder import (
    build_training_experiment_folder,
)
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.profiler_context_manager import ProfilerContextManager
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.experiments.caching import cache_data
from omegaconf import DictConfig

from src.custom_training.custom_training_builder import (
    TrainingEngine,
    build_training_engine,
    update_config_for_training,
)

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()


@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig) -> Optional[TrainingEngine]:
    """
    Main entrypoint for training/validation experiments.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)

    # Configure logger
    build_logger(cfg)

    # Override configs based on setup, and print config
    update_config_for_training(cfg)

    # Create output storage folder
    build_training_experiment_folder(cfg=cfg)

    # Build worker
    worker = build_worker(cfg)

    if cfg.py_func == "train":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            last_ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "last.ckpt")
            if cfg.is_resume_last:
                if os.path.exists(last_ckpt_path):  # 已经生成了自己目录下的last.ckpt
                    engine.trainer.fit(model=engine.model, datamodule=engine.datamodule, ckpt_path=last_ckpt_path)
                elif cfg.checkpoint and os.path.exists(cfg.checkpoint):    # load autoencoder_checkpoint
                    engine.trainer.fit(model=engine.model, datamodule=engine.datamodule, ckpt_path=cfg.checkpoint)
                else:   # 重新训练
                    engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
            else:
                engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    elif cfg.py_func == "fintune_planning":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # 加载pretrain model
        if cfg.checkpoint and os.path.exists(cfg.checkpoint):    # load autoencoder_checkpoint
            pretrained_ckpt = torch.load(cfg.checkpoint)['state_dict']
            engine.model.load_state_dict(pretrained_ckpt, strict=False)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            last_ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "last.ckpt")
            if cfg.is_resume_last and os.path.exists(last_ckpt_path):
                engine.trainer.fit(model=engine.model, datamodule=engine.datamodule, ckpt_path=last_ckpt_path)
            else:
                engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    if cfg.py_func == "validate":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "validate"):
            engine.trainer.validate(
                model=engine.model,
                datamodule=engine.datamodule,
                ckpt_path=cfg.checkpoint,
            )
        return engine
    elif cfg.py_func == "test":
        # Build training engine
        with ProfilerContextManager(
            cfg.output_dir, cfg.enable_profiling, "build_training_engine"
        ):
            engine = build_training_engine(cfg, worker)

        # Test model
        logger.info("Starting testing...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "testing"):

            engine.trainer.test(
                model=engine.model, 
                datamodule=engine.datamodule, 
                ckpt_path=cfg.checkpoint
            )
        return engine
    elif cfg.py_func == "cache":
        # Precompute and cache all features
        logger.info("Starting caching...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "caching"):
            cache_data(cfg=cfg, worker=worker)
        return None
    elif cfg.py_func == "train_map_tokenizer":
        # Build training engine
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "build_training_engine"):
            engine = build_training_engine(cfg, worker)

        # Run training
        logger.info("Starting training...")
        with ProfilerContextManager(cfg.output_dir, cfg.enable_profiling, "training"):
            last_ckpt_path = os.path.join(cfg.output_dir, "best_model", "last.ckpt")
            if cfg.is_resume_last:
                if os.path.exists(last_ckpt_path):  # 已经生成了自己目录下的last.ckpt
                    engine.trainer.fit(model=engine.model, datamodule=engine.datamodule, ckpt_path=last_ckpt_path)
                elif cfg.autoencoder_checkpoint:    # load autoencoder_checkpoint
                    engine.trainer.fit(model=engine.model, datamodule=engine.datamodule, ckpt_path=cfg.autoencoder_checkpoint)
                else:   # 重新训练
                    engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
            else:
                engine.trainer.fit(model=engine.model, datamodule=engine.datamodule)
        return engine
    else:
        raise NameError(f"Function {cfg.py_func} does not exist")

if __name__ == "__main__":
    main()