from typing import Literal

import torch
from lightning import LightningModule
from lion_pytorch import Lion
from torch import nn
from torch.optim import AdamW

# Note: Removed torchjd imports as they are no longer needed for GLS


class ModelWrapper(LightningModule):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: Literal["AdamW", "Lion"] = "AdamW",
        mtl: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.lrs_config = lrs_config
        # The 'mtl' flag now controls whether to use GLS for loss combination
        self.mtl = mtl

        # Automatic optimization is now always enabled, simplifying the training loop
        # self.automatic_optimization = False # <- This is no longer needed

    def forward(self, inputs):
        return self.model(inputs)

    def predict(self, outputs):
        return self.model.predict(outputs)

    def log_losses(self, losses, stage):
        """
        Logs individual losses and computes the total loss.
        
        If self.mtl is True, it combines losses using the Geometric Loss Strategy (GLS),
        which is the geometric mean of all individual losses. This is numerically
        more stable when implemented as `exp(mean(log(losses)))`.
        
        If self.mtl is False, it simply sums the individual losses.
        """
        individual_losses = []

        # Log the losses from each task and layer, and collect them
        for layer_name, layer_losses in losses.items():
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    self.log(f"{stage}/{layer_name}_{task_name}_{loss_name}", loss_value, sync_dist=True)
                    # Use non-zero losses for the calculation to avoid log(0)
                    if loss_value > 0:
                        individual_losses.append(loss_value)

        # If there are no positive losses, return a zero tensor
        if not individual_losses:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log(f"{stage}/loss", total_loss, sync_dist=True)
            return total_loss

        # Combine losses using GLS if mtl is enabled
        if self.mtl:
            # log-sum-exp trick for numerical stability
            log_loss_values = [torch.log(l) for l in individual_losses]
            total_loss = torch.exp(torch.mean(torch.stack(log_loss_values)))
        else:
            # Default behavior: simple summation
            total_loss = sum(individual_losses)

        # Log the final combined loss
        self.log(f"{stage}/loss", total_loss, sync_dist=True)
        return total_loss

    def log_task_metrics(self, preds, targets, stage):
        # Log any task specific metrics
        for task in self.model.tasks:
            # Check that the task actually has some metrics to log
            if not hasattr(task, "metrics"):
                continue

            # Just log the predictions from the final layer for now
            task_metrics = task.metrics(preds["final"][task.name], targets)

            # If the task returned a non-empty metrics dict, log it
            if task_metrics:
                self.log_dict({f"{stage}/final_{task.name}_{k}": v for k, v in task_metrics.items()})

    def log_metrics(self, preds, targets, stage):
        # First log any task metrics
        self.log_task_metrics(preds, targets, stage)

        # Log any custom metrics implemented by subclass
        if hasattr(self, "log_custom_metrics"):
            self.log_custom_metrics(preds, targets, stage)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # Get the model outputs
        outputs = self.model(inputs)

        # Compute and log losses. log_losses now handles GLS.
        losses = self.model.loss(outputs, targets)
        total_loss = self.log_losses(losses, "train")

        # Get the predictions from the model for logging metrics
        if batch_idx % self.trainer.log_every_n_steps == 0:  # avoid calling predict if possible
            preds = self.predict(outputs)
            self.log_metrics(preds, targets, "train")

        # The manual optimization step is no longer needed.
        # We simply return the calculated GLS loss for Lightning to handle.
        return total_loss

    def validation_step(self, batch):
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        losses = self.model.loss(outputs, targets)
        total_loss = self.log_losses(losses, "val") # log_losses will use GLS if mtl=True

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")

        return total_loss

    def test_step(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)

        # Calculate loss to also run matching
        losses = self.model.loss(outputs, targets)

        # Get the predictions from the model
        preds = self.model.predict(outputs)

        return outputs, preds, losses

    def on_train_start(self):
        # Manually override the learning rate in case we are starting
        # from a checkpoint that had a LRS and now we want a flat LR
        if self.lrs_config.get("skip_scheduler"):
            for optimizer in self.trainer.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lrs_config["initial"]

    def configure_optimizers(self):
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW
        elif self.optimizer.lower() == "lion":
            optimizer = Lion
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        opt = optimizer(self.model.parameters(), lr=self.lrs_config["initial"], weight_decay=self.lrs_config["weight_decay"])

        if not self.lrs_config.get("skip_scheduler"):
            # Configure the learning rate scheduler
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lrs_config["max"],
                total_steps=self.trainer.estimated_stepping_batches,
                div_factor=self.lrs_config["max"] / self.lrs_config["initial"],
                final_div_factor=self.lrs_config["initial"] / self.lrs_config["end"],
                pct_start=float(self.lrs_config["pct_start"]),
            )
            sch = {"scheduler": sch, "interval": "step"}
            return [opt], [sch]
            
        print("Skipping learning rate scheduler.")
        return opt

    # The mlt_opt method is no longer required with GLS and automatic optimization
    # def mlt_opt(self, losses, outputs):
    #     ...