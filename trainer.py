"""
Generic trainer class template.
"""

import os
from datetime import datetime
import torch
import torch.optim as optim
import torch.amp as amp
from torchinfo import summary
import torch.nn.functional as F
from torcheval.metrics.functional import binary_accuracy
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


class ModelTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer: optim.Optimizer,
        scaler: amp.grad_scaler.GradScaler,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        device: torch.device | None = None,
        save_interval=1,
    ):
        """
        Initializes the DL Model, creates the checkpoints, and logs directories.
        Args:
            model (nn.Module): the Pytorch model to train.
            criterion (nn.Module): the loss function.
            optimizer (optim.Optimizer)
            Scheduler (optim.lr_scheduler): learning rate scheduler.
            checkpoint_dir (str): directory to save/load checkpoints. if a string is provided, a directory will be created under ./logs, otherwise logs are saved directly in ./logs
            log_dir (str): directory to save TensorBoard logs. if a string is provided, a directory will be created under ./checkpoints, otherwise logs are saved directly in ./checkpoints
            device (torch.device): Device to run the model on ('cuda', 'cpu', 'mps').
            save_interval (int): the epoch save interval.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.checkpoint_dir = os.path.join("./checkpoints", checkpoint_dir)
        # make a directory per run
        self.log_dir = os.path.join(
            "./logs", log_dir, datetime.now().strftime("%d%m%Y-%H%M%S")
        )
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.save_interval = save_interval

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print("Model summary:")
        summary(
            self.model,
            input_size=(model.d_model, 200),
            batch_dim=0,
        )

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint as: /path/to/checkpoint/ckpt_epoch_<epoch>.pt
        Args:
            epoch (int): the epoch number associated with the checkpoint.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"ckpt_epoch_{epoch:02}.pt")
        torch.save(
            {
                "epoch": epoch,
                "d_model": self.model.d_model,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path=None):
        """
        Loads the last model checkpoint. if checkpoint_path is None, it looks for the last checkpoint in the directory provided when creating the trainer.
        #! Note: this assumes checkpoints are named as: /path/to/checkpoint/ckpt_epoch_<n>.pt, where n is epoch number.
        Returns:
            int: the epoch to resume training from or 0 if no checkpoints were found.
        """
        # no specific checkpoint specified, load last checkpoint.
        if checkpoint_path is None:
            checkpoints = [
                f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")
            ]
            if not checkpoints:
                print("No checkpoints found.")
                return 0
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            last_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(self.checkpoint_dir, last_checkpoint)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.d_model = checkpoint["d_model"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"] is not None
            and self.scaler is not None
        ):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if (
            "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
            and self.scheduler is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"checkpoint loaded from: {checkpoint_path}")
        return checkpoint["epoch"]

    def train_one_epoch(self, data_loader: DataLoader):
        """
        Args:
            data_loader (DataLoader): DataLoader for validation data.
        Returns:
            metrics (Dict): train metrics. Check get_metrics() for the returned metrics.
        """
        self.model.train()
        running_loss = 0.0

        all_y_t = []
        all_y_h = []

        n = len(data_loader)
        for i, (x, y) in enumerate(data_loader, start=1):
            print(f"Step {i}/{n}", end="\r")
            y = y.to(self.device)

            self.optimizer.zero_grad()
            with amp.autocast(device_type=self.device.type, enabled=True):
                logits = self.model(x)
                loss = self.criterion(logits, y.long())

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * y.size(0)
            _, predicted = torch.max(F.softmax(logits, dim=1), dim=1)

            # Collect all outputs and labels
            all_y_t.extend(y.cpu().numpy())
            all_y_h.extend(predicted.cpu().detach().numpy())

        all_y_t = torch.tensor(all_y_t).int().bool()
        all_y_h = torch.tensor(all_y_h)
        avg_loss = running_loss / len(data_loader)
        print()
        return {"Loss": avg_loss, **self.get_metrics(all_y_h, all_y_t)}

    def train(self, num_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model over multiple epochs.
        Args:
            num_epochs (int): number of epochs to train the model for. The last checkpoint is automatically loaded and the model is trained for <num_epochs> more
            train_loader (DataLoader): train DataLoader.
            val_loader (DataLoader): validation DataLoader.
        """
        # create log dir and tensorboard loggers for current training run
        train_log_dir = os.path.join(self.log_dir, "train")
        val_log_dir = os.path.join(self.log_dir, "validation")
        os.makedirs(train_log_dir, exist_ok=True)
        os.makedirs(val_log_dir, exist_ok=True)
        self.train_writer = SummaryWriter(log_dir=train_log_dir)
        self.val_writer = SummaryWriter(log_dir=val_log_dir)

        # load last checkpoint (or start from scratch if 1)
        start_epoch = max(self.load_checkpoint(), 1)
        total_epochs = start_epoch + num_epochs
        # log training info
        print("Starting training from epoch:", start_epoch)
        print(f"Using device: {self.device.type} for training.")
        print(f"Using {train_loader.num_workers} workers for DataLoader")

        prev_loss = float("inf")
        prev_acc = 0
        train_metrics = {}
        val_metrics = {}
        for epoch in range(start_epoch, total_epochs):
            try:
                print(f"Epoch {epoch}/{total_epochs}")
                train_metrics = self.train_one_epoch(train_loader)
                val_metrics = self.validate(val_loader)
                if self.scheduler is not None:
                    self.scheduler.step()  # if using epoch based scheduler
                    # self.scheduler.step(val_metrics['Loss']) # if using reduce on plateau

                # log metrics for the epoch
                for k, v in train_metrics.items():
                    self.train_writer.add_scalar(f"{k}/Epoch", v, epoch)

                for k, v in val_metrics.items():
                    self.val_writer.add_scalar(f"{k}/Epoch", v, epoch)

                # save strategy
                # on loss decrease or accuracy increase, every save_interval epochs or last epoch
                if (
                    (train_metrics["Loss"] < prev_loss)
                    or (val_metrics["Accuracy"] > prev_acc)
                    or epoch % self.save_interval == 0
                    or epoch == num_epochs
                ):
                    self.save_checkpoint(epoch)

                prev_loss = val_metrics["Loss"]
                prev_acc = val_metrics["Accuracy"]

            except KeyboardInterrupt:
                print("Training interrupted. Saving checkpoint and exiting...")
                self.save_checkpoint(epoch)
                break
        else:
            print("Finished training")
            print("Train results:")
            for k, v in train_metrics.items():
                print(f"{k}: {v:.4f}")
            print()
            for k, v in val_metrics.items():
                print(f"{k}: {v:.4f}")
            print()
        self.close()

    def validate(self, data_loader: DataLoader):
        """
        Args:
            data_loader (DataLoader): validation DataLoader.
        Returns:
            metrics (Dict): validation metrics. Check get_metrics() for the returned metrics.
        """
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0

            all_y_t = []
            all_y_h = []

            for x, y in data_loader:
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y.long())

                running_loss += loss.item() * y.size(0)
                _, predicted = torch.max(F.softmax(logits, dim=1), dim=1)

                # Collect all outputs and labels
                all_y_t.extend(y.cpu().numpy())
                all_y_h.extend(predicted.cpu().detach().numpy())

            all_y_t = torch.tensor(all_y_t).int().bool()
            all_y_h = torch.tensor(all_y_h)

            avg_loss = running_loss / len(data_loader)
            return {"Loss": avg_loss, **self.get_metrics(all_y_h, all_y_t)}

    def get_metrics(self, y_h: torch.Tensor, y_t: torch.Tensor):
        return {
            "Accuracy": binary_accuracy(y_h, y_t),
        }

    def test(self, test_loader):
        """
        Evaluate the model on the test data
        Args:
            test_loader (DataLoader): test DataLoader.
        """
        print("Evaluating model...")
        metrics = self.validate(test_loader)
        print("Test results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print()

    def close(self):
        """
        Close the Tensorboard writer.
        """
        self.train_writer.close()
        self.val_writer.close()
