import pytorch_lightning as pl
import torch
from torch import optim
from torchvision import utils as vutils

from slot_attention.model import SlotAttentionModel
from slot_attention.params import SlotAttentionParams
from slot_attention.utils import ClampImage
from slot_attention.utils import Tensor


class SlotAttentionMethod(pl.LightningModule):
    def __init__(self, model: SlotAttentionModel, datamodule: pl.LightningDataModule, params: SlotAttentionParams):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.params = params

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        train_loss = self.model.loss_function(batch)
        logs = {key: val.item() for key, val in train_loss.items()}
        self.log_dict(logs, sync_dist=True)
        return train_loss

    def sample_images(self):
        dl = self.datamodule.val_dataloader()
        batch = next(iter(dl))[: self.params.n_samples]
        if self.params.gpus > 0:
            batch = batch.to(self.device)
        recon_combined, recons, masks, slots = self.model.forward(batch)

        # combine images in a nice way so we can display all outputs in one grid
        out = torch.cat(
            [
                ClampImage()(batch.unsqueeze(1)),  # original images
                ClampImage()(recon_combined.unsqueeze(1)),  # reconstructions
                ClampImage()(recons * masks + (1 - masks)),  # each slot
            ],
            dim=1,
        )

        batch_size, num_slots, C, H, W = recons.shape
        images = vutils.make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        # if self.params.empty_cache:
        #     torch.cuda.empty_cache()
        return images

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        val_loss = self.model.loss_function(batch)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {
            "avg_val_loss": avg_loss,
        }
        self.log_dict(logs, sync_dist=True)
        print("; ".join([f"{k}: {v.item():.6f}" for k, v in logs.items()]))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params.scheduler_gamma)
        return [optimizer], [scheduler]
