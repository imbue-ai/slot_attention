import json
from typing import Optional

import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms

from slot_attention.data import CLEVRDataModule
from slot_attention.method import SlotAttentionMethod
from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ImageLogCallback
from slot_attention.utils import rescale


def main(params: Optional[SlotAttentionParams] = None):
    if params is None:
        params = SlotAttentionParams()

    assert params.num_slots > 1, "Must have at least 2 slots."

    if params.is_verbose:
        print(f"INFO: limiting the dataset to only images with `num_slots - 1` ({params.num_slots - 1}) objects.")
        if params.num_train_images:
            print(f"INFO: restricting the train dataset size to `num_train_images`: {params.num_train_images}")
        if params.num_val_images:
            print(f"INFO: restricting the validation dataset size to `num_val_images`: {params.num_val_images}")

    clevr_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(rescale),  # rescale between -1 and 1
            transforms.Resize(params.resolution),
        ]
    )

    clevr_datamodule = CLEVRDataModule(
        data_root=params.data_root,
        max_n_objects=params.num_slots - 1,
        train_batch_size=params.batch_size,
        val_batch_size=params.val_batch_size,
        clevr_transforms=clevr_transforms,
        num_train_images=params.num_train_images,
        num_val_images=params.num_val_images,
        num_workers=params.num_workers,
    )

    model = SlotAttentionModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        empty_cache=params.empty_cache,
    )

    method = SlotAttentionMethod(model=model, datamodule=clevr_datamodule, params=params)

    logger_name = "slot-attention-clevr6"
    logger = pl_loggers.WandbLogger(project="slot-attention-clevr6", name=logger_name)
    model_checkpoint = ModelCheckpoint(
        dirpath="./best_checkpoints",
        monitor="avg_val_loss",
        filename="slot-attention-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        logger=logger if params.is_logger_enabled else False,
        accelerator="ddp" if params.gpus > 1 else None,
        num_sanity_val_steps=params.num_sanity_val_steps,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        callbacks=[model_checkpoint, LearningRateMonitor("step"), ImageLogCallback(),]
        if params.is_logger_enabled
        else [],
    )
    trainer.fit(method)

    json.dump(
        {
            "best_model_path": model_checkpoint.best_model_path,
            "best_model_score": model_checkpoint.best_model_score.item()
            if model_checkpoint.best_model_score
            else None,
        },
        open("checkpoint_details.json", "w"),
    )


if __name__ == "__main__":
    main()
