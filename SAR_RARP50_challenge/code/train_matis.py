import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from data_loader import create_data_loaders
from monai.metrics import compute_iou, compute_generalized_dice
from model.MATIS import MATIS
from callback import EarlyStopping


class ModelTrainer:
    """
    Class to manage the training process of a model for semantic segmentation.

    Parameters:
        train_csv_file (str): File path for the training dataset CSV.
        test_csv_file (str): File path for the testing dataset CSV.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training and evaluation.
    """

    def __init__(self, train_csv_file, test_csv_file, num_epochs=1000, batch_size=4):
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = MATIS()
        self.model.to(self.device)
        self.model = nn.DataParallel(self.model, device_ids=[1, 0])
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.1, patience=3, verbose=True)
        self.early_stopping = EarlyStopping(patience=50, verbose=True, mode="max")

        # Metrics to track
        self.metrics = {
            "epoch": [], "learning_rate": [], "train_loss": [], "test_loss": [],
            "train_miou": [], "test_miou": [],
            "train_mnsd": [], "test_mnsd": [],
            "train_seg_score": [], "test_seg_score": []
        }

    def train(self):
        train_loader, test_loader = create_data_loaders(self.train_csv_file, self.test_csv_file, self.batch_size)

        for epoch in range(self.num_epochs):
            self.run_training_epoch(train_loader)
            test_results = self.evaluate(test_loader)

            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Train Loss: {self.metrics['train_loss'][-1]:.4f}, "
                  f"Test Loss: {test_results['loss']:.4f}, "
                  f"Train mIoU: {self.metrics['train_miou'][-1]:.4f}, "
                  f"Test mIoU: {test_results['miou']:.4f}, "
                  f"Train NSD: {self.metrics['train_mnsd'][-1]:.4f}, "
                  f"Test NSD: {test_results['mnsd']:.4f}, "
                  f"Train Seg Score: {self.metrics['train_seg_score'][-1]:.4f}, "
                  f"Test Seg Score: {test_results['seg_score']:.4f}")

            self.metrics["epoch"].append(epoch + 1)
            self.scheduler.step(test_results["seg_score"])

            if self.early_stopping.early_stop:
                print("Stopping training early due to early stopping criteria.")
                break

        self.save_results()

    def run_training_epoch(self, train_loader):
        self.model.train()
        total_loss, total_miou, total_mnsd, total_seg_score = 0, 0, 0, 0

        for images, masks in train_loader:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = self.model(images)
                outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
                loss = self.criterion(outputs, masks.squeeze(1).long())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            miou, mnsd, seg_score = self.calculate_metrics(outputs, masks)
            total_miou += miou
            total_mnsd += mnsd
            total_seg_score += seg_score

        self.record_metrics(total_loss, len(train_loader), total_miou, total_mnsd, total_seg_score, train=True)

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss, total_miou, total_mnsd, total_seg_score = 0, 0, 0, 0

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode="bilinear", align_corners=False)
                loss = self.criterion(outputs, masks.squeeze(1).long())

                total_loss += loss.item()
                miou, mnsd, seg_score = self.calculate_metrics(outputs, masks)
                total_miou += miou
                total_mnsd += mnsd
                total_seg_score += seg_score

        self.record_metrics(total_loss, len(test_loader), total_miou, total_mnsd, total_seg_score, train=False)
        return {"loss": total_loss / len(test_loader), "miou": total_miou / len(test_loader),
                "mnsd": total_mnsd / len(test_loader), "seg_score": total_seg_score / len(test_loader)}

    def calculate_metrics(self, outputs, masks):
        """
        Compute mIoU_avg, NSD_avg, and Final Score.
        """
        # Convert logits to discrete class predictions
        preds = torch.argmax(outputs, dim=1)  # Shape: [Batch, H, W]

        # Squeeze ground truth to match predictions
        masks = masks.squeeze(1)  # Shape: [Batch, H, W]

        # Ensure predictions and ground truth have matching shapes
        if preds.shape != masks.shape:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, masks={masks.shape}")

        # Compute metrics
        miou = compute_iou(preds, masks, include_background=False).mean().item()
        nsd = compute_generalized_dice(preds.unsqueeze(1), masks.unsqueeze(1)).mean().item()
        seg_score = miou * nsd

        return miou, nsd, seg_score

    def record_metrics(self, total_loss, num_items, total_miou, total_mnsd, total_seg_score, train=True):
        prefix = "train" if train else "test"
        self.metrics[f"{prefix}_loss"].append(total_loss / num_items)
        self.metrics[f"{prefix}_miou"].append(total_miou / num_items)
        self.metrics[f"{prefix}_mnsd"].append(total_mnsd / num_items)
        self.metrics[f"{prefix}_seg_score"].append(total_seg_score / num_items)
        if train:
            self.metrics["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

    def save_results(self):
        df = pd.DataFrame(self.metrics)
        df.to_csv("training_metrics.csv", index=False)
        torch.save(self.model.state_dict(), "unet_model.pth")
        print("Training completed and metrics saved to 'training_metrics.csv'")


if __name__ == "__main__":
    trainer = ModelTrainer("C:/Users/User/Desktop/Shahbaz_project/SAR_RARP50_challenge/Database/train_data.csv",
                           "C:/Users/User/Desktop/Shahbaz_project/SAR_RARP50_challenge/Database/test_data.csv")
    trainer.train()


