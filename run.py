import argparse

import lightning.pytorch as pl
import torch
import torch.optim as optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import ViTWithTokenDropout

import datasets
import datasets.voc
from models.vision_transformers import VisionTransformers

TARGET_TASK_MAP = {
    "eurosat": "single-label",
    "dfc": "multi-label",
    "ade": "multi-label",
    "voc": "multi-label",
    "coco": "multi-label",
    "cityscapes": "multi-label",
    "endotect": "single-label",
    "word": "multi-label",
    "isic": "single-label",
    "bigearthnet": "multi-label",
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_batch_size",
        default=32,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Total batch size for eval."
    )
    parser.add_argument("--num_devices", default=1, type=int, help="Number of GPU")
    parser.add_argument("--device", default="gpu", type=str, help="gpu or cpu")
    parser.add_argument(
        "--learning_rate",
        default=0.03,
        type=float,
        help="The initial learning rate for SGD.",
    )
    parser.add_argument(
        "--num_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--depth", default=12, type=int, help="Total number of blocks.")
    parser.add_argument("--patch_size", default=12, type=int, help="Patch Size.")
    parser.add_argument("--num_classes", default=8, type=int, help="Number of classes")
    parser.add_argument(
        "--num_classes_ft", default=8, type=int, help="Number of classes finetuned"
    )
    parser.add_argument("--num_heads", default=16, type=int, help="Number of heads")
    parser.add_argument(
        "--accumulate_grad",
        default=1,
        type=int,
        help="Accumulate gradient every N batches",
    )
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument(
        "--num_channels", default=13, type=int, help="Number of channels"
    )
    parser.add_argument("--dataset", default="dfc", type=str, help="Dataset Selected")
    parser.add_argument("--exp_name", default="", type=str, help="Special Exp. Name")
    parser.add_argument("--opt", default="adamw", type=str, help="Optimizer")
    parser.add_argument("--mult", default=1, type=int, help="Multiplication Factor")
    parser.add_argument("--warmup", default=10, type=int, help="Warmup epochs")
    parser.add_argument(
        "--pr_rate", default=0.2, type=float, help="Pruning coefficient"
    )
    parser.add_argument("--dp_rate", default=0.5, type=float, help="Dropout Token Rate")
    parser.add_argument(
        "--diversity_coeff",
        default=0.1,
        type=float,
        help="Diversity penalty coefficient",
    )
    parser.add_argument(
        "--arch",
        default="vit",
        type=str,
        help="Architecture desired - Vit, DeepViT etc.",
    )
    parser.add_argument("--imgsize", nargs="+", type=int)
    parser.add_argument("--diversify", action="store_true")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--multimodal", action="store_true")
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--randomly_mask", action="store_true")
    parser.add_argument("--logger", action="store_true")
    parser.add_argument("--adaptive_dp_rate", action="store_true")
    return parser.parse_args()


def setup_logger(
    dataset,
    arch,
    patch_size,
    batch_size,
    depth,
    learning_rate,
    prune,
    dp_rate,
    pr_rate,
    exp_name,
    logger,
):
    """Create and configure a wandb logger for tracking experiments.

    Parameters
    ----------
    dataset : Name of the dataset to use
    arch : Model architecture identifier
    patch_size : Patch size used to tokenize the input images.
    batch_size : Mini-batch size used for training and validation.
    depth : Number of transformer layers / blocks in the encoder.
    learning_rate : Learning rate used
    prune : Whether to enable parameter pruning of the model.
    dp_rate : [CLS] token dropout rate (masking ratio)
    pr_rate : Pruning parameter (lambda value)
    exp_name : Experiment name or suffix used for logging and checkpoints.
    logger : If True, enable experiment tracking with wandb."""

    run_name = "{}_arch_{}_patch_{}_bs_{}_depth_{}_lr_{}_prune_{}_pr_rate_{}_dp_rate_{}_{}".format(
        dataset,
        arch,
        patch_size,
        batch_size,
        depth,
        learning_rate,
        prune,
        dp_rate,
        pr_rate,
        exp_name,
    )
    entity_name = ""
    project_name = ""
    if logger:
        mode = "online"
    else:
        mode = "disabled"
    wandb = WandbLogger(
        project=project_name, entity=entity_name, name=run_name, mode=mode
    )
    return wandb


def setup_model(
    num_classes,
    patch_size,
    prune,
    num_channels,
    dp_rate,
):
    """
    num_classes : Number of output classes for the classification head.
    patch_size : Patch size used to tokenize the input images.
    prune : Whether to enable parameter pruning of the model.
    dp_rate : [CLS] token dropout rate (masking ratio)
    """

    model = ViTWithTokenDropout(
        patch_size=patch_size,
        prune=prune,
        in_chans=num_channels,
        num_classes=num_classes,
        rate=dp_rate,
    )

    return model


def create_dataset(
    dataset, mult, imgsize, train_batch_size, eval_batch_size, randomly_mask=False
):
    """Create and initialize the Lightning data module corresponding to the dataset"""
    print(f"Setting up the {dataset} dataset")
    if dataset == "dfc":
        data_module = datasets.dfc.DFCDataModule()
        data_module.setup(mult, train_batch_size, eval_batch_size)
    elif dataset == "endotect":
        data_module = datasets.endotect.EndotectDataModule()
        data_module.setup(train_batch_size, eval_batch_size)
    elif dataset == "ade":
        data_module = datasets.ade.Ade20KDataModule()
        data_module.setup(train_batch_size, eval_batch_size)
    elif dataset == "cityscapes":
        data_module = datasets.cityscapes.CityscapesDataModule()
        if randomly_mask:
            data_module.setup(
                mult, imgsize, train_batch_size, eval_batch_size, randomly_mask=True
            )
        else:
            data_module.setup(mult, imgsize, train_batch_size, eval_batch_size)
    elif dataset == "isic":
        data_module = datasets.isic.ISICDataModule()
        data_module.setup(train_batch_size, eval_batch_size)
    elif dataset == "voc":
        data_module = datasets.voc.VOCDataModule()
        data_module.setup(train_batch_size, eval_batch_size)
    elif dataset == "coco":
        data_module = datasets.coco.COCODataModule()
        data_module.setup(train_batch_size, eval_batch_size)
    else:
        print("The dataset doesn't exist.")
        return
    return data_module


def setup_criterion_optimizer_scheduler(
    dataset, opt, learning_rate, lr_scheduler, model
):
    """Construct the loss function, optimizer, and optional learning rate scheduler.
    The choice of loss depends on whether the task is single-label or multi-label."""
    if TARGET_TASK_MAP[dataset] == "single-label":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif TARGET_TASK_MAP[dataset] == "multi-label":
        criterion = torch.nn.BCELoss(reduction="mean").to(device)
    else:
        raise ValueError("Invalid target specified")
    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=learning_rate)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer specified")
    scheduler = None
    if lr_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    return (criterion, optimizer, scheduler)


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    wandb_logger = setup_logger(
        args.dataset,
        args.arch,
        args.patch_size,
        args.train_batch_size,
        args.depth,
        args.learning_rate,
        args.prune,
        args.dp_rate,
        args.pr_rate,
        args.exp_name,
        args.logger,
    )
    model = setup_model(
        args.num_classes,
        args.patch_size,
        args.prune,
        args.num_channels,
        args.dp_rate,
    )
    criterion, optimizer, scheduler = setup_criterion_optimizer_scheduler(
        args.dataset, args.opt, args.learning_rate, args.lr_scheduler, model
    )
    model_module = VisionTransformers(
        model,
        criterion,
        optimizer,
        scheduler,
        args.num_classes,
        args.num_heads,
        args.train_batch_size,
        args.diversity_coeff,
        args.prune,
        args.diversify,
        args.lr_scheduler,
        TARGET_TASK_MAP[args.dataset],
        args.multimodal,
        args.adaptive_dp_rate,
    )
    data_module = create_dataset(
        args.dataset,
        args.mult,
        args.imgsize,
        args.train_batch_size,
        args.eval_batch_size,
        args.randomly_mask,
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        filename="{args.dataset}-{epoch:02d}-{val_accuracy:.2f}",
    )
    trainer = pl.Trainer(
        accelerator=args.device,
        devices=args.num_devices,
        max_epochs=args.num_epochs,
        log_every_n_steps=args.accumulate_grad,
        accumulate_grad_batches=args.accumulate_grad,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(
            {
                "learning_rate": args.learning_rate,
                "epochs": args.num_epochs,
                "batch_size": args.train_batch_size,
                "depth": args.depth,
                "patch_size": args.patch_size,
                "arch": args.arch,
            }
        )
    trainer.fit(
        model_module, data_module.train_dataloader(), data_module.val_dataloader()
    )


if __name__ == "__main__":
    main()
