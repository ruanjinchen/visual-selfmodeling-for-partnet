import os
import sys
import yaml
import torch
import pprint
from munch import munchify
from models import VisModelingModel

# ---- Lightning 2.x 首选导入；若不兼容则退回到旧别名，原来的版本太老了 ----
try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from lightning.pytorch.strategies import DDPStrategy
except Exception:
    # 兼容
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    try:
        from pytorch_lightning.strategies import DDPStrategy
    except Exception:
        DDPStrategy = None  # 单卡时不需要

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if getattr(cfg, "if_cuda", False):
        torch.cuda.manual_seed(cfg.seed)

def build_model(cfg, log_dir):
    return VisModelingModel(
        lr=cfg.lr,
        seed=cfg.seed,
        dof=cfg.dof,
        if_cuda=cfg.if_cuda,
        if_test=False,
        gamma=cfg.gamma,
        log_dir=log_dir,
        train_batch=cfg.train_batch,
        val_batch=cfg.val_batch,
        test_batch=cfg.test_batch,
        num_workers=cfg.num_workers,
        model_name=cfg.model_name,
        data_filepath=cfg.data_filepath,
        loss_type=cfg.loss_type,
        coord_system=cfg.coord_system,
        lr_schedule=cfg.lr_schedule
    )

def build_trainer(cfg, log_dir, with_ckpt=False):
    # Lightning 2.x: 用 accelerator/devices/strategy
    num_gpus = int(getattr(cfg, "num_gpus", 0) or 0)
    use_gpu = num_gpus > 0
    accelerator = "gpu" if use_gpu else "cpu"
    devices = num_gpus if use_gpu else 1

    # 多卡时才启用 DDPStrategy；单卡设为 "auto"
    strategy = "auto"
    if use_gpu and devices > 1 and DDPStrategy is not None:
        strategy = DDPStrategy(find_unused_parameters=False)

    callbacks = []
    if with_ckpt:
        ckpt_cb = ModelCheckpoint(
            dirpath=log_dir,
            filename="{epoch}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
        callbacks.append(ckpt_cb)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.epochs,
        deterministic=True,
        strategy=strategy,
        default_root_dir=log_dir,
        callbacks=callbacks,
        # precision 可以按需改为 "16-mixed" 以启用 AMP；默认 32 位
        # precision="16-mixed",
    )
    return trainer

def main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir, cfg.model_name, cfg.tag, str(cfg.seed)])
    model = build_model(cfg, log_dir)
    trainer = build_trainer(cfg, log_dir, with_ckpt=False)
    trainer.fit(model)

def main_kinematic():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir, cfg.model_name, cfg.tag, str(cfg.seed)])
    model = build_model(cfg, log_dir)

    # 回调：保存 val_loss 最优
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    trainer = Trainer(
        accelerator=("gpu" if int(getattr(cfg, "num_gpus", 0) or 0) > 0 else "cpu"),
        devices=(int(getattr(cfg, "num_gpus", 0) or 0) if int(getattr(cfg, "num_gpus", 0) or 0) > 0 else 1),
        max_epochs=cfg.epochs,
        deterministic=True,
        strategy=(DDPStrategy(find_unused_parameters=False)
                  if int(getattr(cfg, "num_gpus", 0) or 0) > 1 and DDPStrategy is not None else "auto"),
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
        # check_val_every_n_epoch=1  # 每个 epoch 验证一次（默认即如此）
    )

    # 从命令行第 3 个参数加载预训练编码器（保持原逻辑）
    if len(sys.argv) >= 4:
        model.extract_kinematic_encoder_model(sys.argv[3])
    trainer.fit(model)

def main_kinematic_scratch():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir, cfg.model_name, cfg.tag, str(cfg.seed)])
    model = build_model(cfg, log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    trainer = Trainer(
        accelerator=("gpu" if int(getattr(cfg, "num_gpus", 0) or 0) > 0 else "cpu"),
        devices=(int(getattr(cfg, "num_gpus", 0) or 0) if int(getattr(cfg, "num_gpus", 0) or 0) > 0 else 1),
        max_epochs=cfg.epochs,
        deterministic=True,
        strategy=(DDPStrategy(find_unused_parameters=False)
                  if int(getattr(cfg, "num_gpus", 0) or 0) > 1 and DDPStrategy is not None else "auto"),
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model)

if __name__ == '__main__':
    # 兼容：以前要求第二个参数必须写 NA/kinematic/kinematic-scratch
    mode = sys.argv[2] if len(sys.argv) >= 3 else "NA"
    if mode == 'kinematic':
        main_kinematic()
    elif mode == 'kinematic-scratch':
        main_kinematic_scratch()
    else:
        main()
