# #
# # import torch
# # import torch.optim as optim
# # import torch.nn as nn
# # from torch.cuda.amp import GradScaler, autocast
# # from torch.optim.lr_scheduler import ReduceLROnPlateau
# # import pandas as pd
# # from data_loader import create_data_loaders
# # from metrics import SegmentationMetrics
# # from model.unet import UNet  # Ensure 'unet.py' contains the UNet model definition
# # from callback import EarlyStopping
# #
# # seg_metrics = SegmentationMetrics()
# #
# #
# # class ModelTrainer:
# #     """
# #     Class to manage the training process of a UNet model for semantic segmentation.
# #
# #     Parameters:
# #         train_csv_file (str): File path for the training dataset CSV.
# #         test_csv_file (str): File path for the testing dataset CSV.
# #         num_epochs (int): Number of epochs to train the model.
# #         batch_size (int): Batch size for training and evaluation.
# #     """
# #
# #     def __init__(self, train_csv_file, test_csv_file, num_epochs=1000, batch_size=32):
# #         self.train_csv_file = train_csv_file
# #         self.test_csv_file = test_csv_file
# #         self.num_epochs = num_epochs
# #         self.batch_size = batch_size
# #         self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# #         self.model = UNet(n_channels=3, n_classes=10)
# #         self.model = nn.DataParallel(self.model)
# #         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
# #         self.criterion = nn.CrossEntropyLoss()
# #         self.scaler = GradScaler()
# #         self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=3, verbose=True)
# #         self.early_stopping = EarlyStopping(patience=50, verbose=True, mode='max')
# #         self.metrics = {
# #             'epoch': [], 'learning_rate': [], 'train_loss': [], 'test_loss': [],
# #             'train_iou': [], 'test_iou': [], 'train_dice': [], 'test_dice': [],
# #             'train_miou': [], 'test_miou': [], 'train_mnsd': [], 'test_mnsd': [],
# #             'train_seg_score': [], 'test_seg_score': []
# #         }
# #
# #     def train(self):
# #         """Execute the training loop, evaluate and save the model, and metrics upon completion."""
# #         train_loader, test_loader = create_data_loaders(self.train_csv_file, self.test_csv_file, self.batch_size)
# #         print(train_loader)
# #         print(self.device)
# #
# #         for epoch in range(self.num_epochs):
# #             self.run_training_epoch(train_loader)
# #             test_results = self.evaluate(test_loader)
# #
# #             print(f'Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {self.metrics["train_loss"][-1]:.4f}, '
# #                   f'Train IoU: {self.metrics["train_iou"][-1]:.4f}, Train Dice: {self.metrics["train_dice"][-1]:.4f}, '
# #                   f'Test Loss: {test_results["loss"]:.4f}, '
# #                   f'Test IoU: {test_results["iou"]:.4f}, Test Dice: {test_results["dice"]:.4f}')
# #             print(f'Epoch {epoch + 1}/{self.num_epochs} - Train mIoU: {self.metrics["train_miou"][-1]:.4f}, '
# #                   f'Train mNSD: {self.metrics["train_mnsd"][-1]:.4f}, Train Segmentation Score: {self.metrics["train_seg_score"][-1]:.4f}, '
# #                   f'Test mIoU: {test_results["miou"]:.4f}, '
# #                   f'Test mNSD: {test_results["mnsd"]:.4f}, Test Segmentation Score: {test_results["seg_score"]:.4f}')
# #             self.metrics['epoch'].append(epoch + 1)
# #             self.scheduler.step(test_results['seg_score'])
# #             self.early_stopping(self.metrics['test_seg_score'][-1])
# #             if self.early_stopping.early_stop:
# #                 print("Stopping training early due to early stopping criteria.")
# #                 break
# #
# #         self.save_results()
# #
# #     def run_training_epoch(self, train_loader):
# #         """Run one training epoch and calculate metrics."""
# #         self.model.train()
# #         total_loss, total_iou, total_dice, total_miou, total_mnsd, total_seg_score = 0, 0, 0, 0, 0, 0
# #
# #         for images, masks in train_loader:
# #             images, masks = images.to(self.device), masks.to(self.device)
# #             self.optimizer.zero_grad()
# #
# #             with autocast():
# #                 outputs = self.model(images)
# #                 loss = self.criterion(outputs, masks.squeeze(1).long())
# #
# #             self.scaler.scale(loss).backward()
# #             self.scaler.step(self.optimizer)
# #             self.scaler.update()
# #
# #             total_loss += loss.item()
# #             preds, labels = seg_metrics.convert_to_boolean_tensors(outputs, masks)
# #             total_iou += seg_metrics.calculate_iou(preds, labels).item()
# #             total_dice += seg_metrics.calculate_dice(preds, labels).item()
# #
# #             # Additional metrics
# #             distance_maps = [seg_metrics.compute_distance_map(label) for label in labels]
# #             total_miou += seg_metrics.calculate_mean_iou([preds], [labels]).item()
# #             total_mnsd += seg_metrics.calculate_mean_nsd([preds], [labels], distance_maps).item()
# #             total_seg_score += seg_metrics.calculate_segmentation_score(total_miou, total_mnsd)
# #
# #         self.record_metrics(total_loss, total_iou, total_dice, len(train_loader), total_miou, total_mnsd, total_seg_score, train=True)
# #
# #     def evaluate(self, test_loader):
# #         """Evaluate the model on the test dataset and return the metrics."""
# #         self.model.eval()
# #         test_loss, test_iou, test_dice, test_miou, test_mnsd, test_seg_score = 0, 0, 0, 0, 0, 0
# #
# #         with torch.no_grad():
# #             for images, masks in test_loader:
# #                 images, masks = images.to(self.device), masks.to(self.device)
# #                 outputs = self.model(images)
# #                 loss = self.criterion(outputs, masks.squeeze(1).long())
# #
# #                 test_loss += loss.item()
# #                 preds, labels = seg_metrics.convert_to_boolean_tensors(outputs, masks)
# #                 test_iou += seg_metrics.calculate_iou(preds, labels).item()
# #                 test_dice += seg_metrics.calculate_dice(preds, labels).item()
# #
# #                 # Additional metrics
# #                 distance_maps = [seg_metrics.compute_distance_map(label) for label in labels]
# #                 test_miou += seg_metrics.calculate_mean_iou([preds], [labels]).item()
# #                 test_mnsd += seg_metrics.calculate_mean_nsd([preds], [labels], distance_maps).item()
# #                 test_seg_score += seg_metrics.calculate_segmentation_score(test_miou, test_mnsd)
# #
# #         self.record_metrics(test_loss, test_iou, test_dice, len(test_loader), test_miou, test_mnsd, test_seg_score, train=False)
# #         self.scheduler.step(test_loss / len(test_loader))
# #         return {'loss': test_loss / len(test_loader), 'iou': test_iou / len(test_loader), 'dice': test_dice / len(test_loader),
# #                 'miou': test_miou / len(test_loader), 'mnsd': test_mnsd / len(test_loader), 'seg_score': test_seg_score / len(test_loader)}
# #
# #     def record_metrics(self, total_loss, total_iou, total_dice, num_items, total_miou, total_mnsd, total_seg_score, train=True):
# #         """Record training or testing metrics."""
# #         prefix = 'train' if train else 'test'
# #         self.metrics[f'{prefix}_loss'].append(total_loss / num_items)
# #         self.metrics[f'{prefix}_iou'].append(total_iou / num_items)
# #         self.metrics[f'{prefix}_dice'].append(total_dice / num_items)
# #         self.metrics[f'{prefix}_miou'].append(total_miou / num_items)
# #         self.metrics[f'{prefix}_mnsd'].append(total_mnsd / num_items)
# #         self.metrics[f'{prefix}_seg_score'].append(total_seg_score / num_items)
# #         if train:
# #             self.metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
# #
# #     def save_results(self):
# #         """Save the trained model and metrics to files."""
# #         df = pd.DataFrame(self.metrics)
# #         df.to_csv('training_metrics.csv', index=False)
# #         torch.save(self.model.state_dict(), 'unet_model.pth')
# #         print("Training completed and metrics saved to 'training_metrics.csv'")
# #
# #
# # if __name__ == "__main__":
# #     trainer = ModelTrainer("C:/Users/User/Desktop/Shahbaz_project/SAR_RARP50_challenge/Database/train_data.csv",
# #                            "C:/Users/User/Desktop/Shahbaz_project/SAR_RARP50_challenge/Database/test_data.csv")
# #
# #     trainer.train()
#
#
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.amp import GradScaler, autocast  # Updated import for autocast and GradScaler
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import pandas as pd
# from data_loader import create_data_loaders
# from metrics import SegmentationMetrics
# from model.unet import UNet  # Ensure 'unet.py' contains the UNet model definition
# from callback import EarlyStopping
#
# seg_metrics = SegmentationMetrics()
#
# class ModelTrainer:
#     """
#     Class to manage the training process of a UNet model for semantic segmentation.
#
#     Parameters:
#         train_csv_file (str): File path for the training dataset CSV.
#         test_csv_file (str): File path for the testing dataset CSV.
#         num_epochs (int): Number of epochs to train the model.
#         batch_size (int): Batch size for training and evaluation.
#     """
#
#     def __init__(self, train_csv_file, test_csv_file, num_epochs=1000, batch_size=64):
#         self.train_csv_file = train_csv_file
#         self.test_csv_file = test_csv_file
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#         self.model = UNet(n_channels=3, n_classes=10)
#         self.model.to(self.device)  # Ensure model is moved to the correct device
#         self.model = nn.DataParallel(self.model, device_ids=[1, 0])  # Specify device_ids to handle the warning
#         self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
#         self.criterion = nn.CrossEntropyLoss()
#         self.scaler = GradScaler()  # Updated usage for GradScaler
#         self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=3, verbose=True)
#         self.early_stopping = EarlyStopping(patience=50, verbose=True, mode='max')
#         self.metrics = {
#             'epoch': [], 'learning_rate': [], 'train_loss': [], 'test_loss': [],
#             'train_iou': [], 'test_iou': [], 'train_dice': [], 'test_dice': [],
#             'train_miou': [], 'test_miou': [], 'train_mnsd': [], 'test_mnsd': [],
#             'train_seg_score': [], 'test_seg_score': []
#         }
#
#     def train(self):
#         train_loader, test_loader = create_data_loaders(self.train_csv_file, self.test_csv_file, self.batch_size)
#         print(train_loader)
#         print(self.device)
#
#         for epoch in range(self.num_epochs):
#             self.run_training_epoch(train_loader)
#             test_results = self.evaluate(test_loader)
#
#             print(f'Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {self.metrics["train_loss"][-1]:.4f}, '
#                   f'Train IoU: {self.metrics["train_iou"][-1]:.4f}, Train Dice: {self.metrics["train_dice"][-1]:.4f}, '
#                   f'Test Loss: {test_results["loss"]:.4f}, '
#                   f'Test IoU: {test_results["iou"]:.4f}, Test Dice: {test_results["dice"]:.4f}')
#             print(f'Epoch {epoch + 1}/{self.num_epochs} - Train mIoU: {self.metrics["train_miou"][-1]:.4f}, '
#                   f'Train mNSD: {self.metrics["train_mnsd"][-1]:.4f}, Train Segmentation Score: {self.metrics["train_seg_score"][-1]:.4f}, '
#                   f'Test mIoU: {test_results["miou"]:.4f}, '
#                   f'Test mNSD: {test_results["mnsd"]:.4f}, Test Segmentation Score: {test_results["seg_score"]:.4f}')
#             self.metrics['epoch'].append(epoch + 1)
#             self.scheduler.step(test_results['seg_score'])
#             self.early_stopping(self.metrics['test_seg_score'][-1])
#             if self.early_stopping.early_stop:
#                 print("Stopping training early due to early stopping criteria.")
#                 break
#
#         self.save_results()
#
#     def run_training_epoch(self, train_loader):
#         self.model.train()
#         total_loss, total_iou, total_dice, total_miou, total_mnsd, total_seg_score = 0, 0, 0, 0, 0, 0
#
#         for images, masks in train_loader:
#             images, masks = images.to(self.device), masks.to(self.device)
#             self.optimizer.zero_grad()
#
#             with autocast(device_type='cuda'):  # Updated usage for autocast
#                 outputs = self.model(images)
#                 loss = self.criterion(outputs, masks.squeeze(1).long())
#
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#
#             total_loss += loss.item()
#             preds, labels = seg_metrics.convert_to_boolean_tensors(outputs, masks)
#             total_iou += seg_metrics.calculate_iou(preds, labels).item()
#             total_dice += seg_metrics.calculate_dice(preds, labels).item()
#
#             # Additional metrics
#             distance_maps = [seg_metrics.compute_distance_map(label) for label in labels]
#             total_miou += seg_metrics.calculate_mean_iou([preds], [labels]).item()
#             total_mnsd += seg_metrics.calculate_mean_nsd([preds], [labels], distance_maps).item()
#             total_seg_score += seg_metrics.calculate_segmentation_score(total_miou, total_mnsd)
#
#         self.record_metrics(total_loss, total_iou, total_dice, len(train_loader), total_miou, total_mnsd, total_seg_score, train=True)
#
#     def evaluate(self, test_loader):
#         self.model.eval()
#         test_loss, test_iou, test_dice, test_miou, test_mnsd, test_seg_score = 0, 0, 0, 0, 0, 0
#
#         with torch.no_grad():
#             for images, masks in test_loader:
#                 images, masks = images.to(self.device), masks.to(self.device)
#                 outputs = self.model(images)
#                 loss = self.criterion(outputs, masks.squeeze(1).long())
#
#                 test_loss += loss.item()
#                 preds, labels = seg_metrics.convert_to_boolean_tensors(outputs, masks)
#                 test_iou += seg_metrics.calculate_iou(preds, labels).item()
#                 test_dice += seg_metrics.calculate_dice(preds, labels).item()
#
#                 # Additional metrics
#                 distance_maps = [seg_metrics.compute_distance_map(label) for label in labels]
#                 test_miou += seg_metrics.calculate_mean_iou([preds], [labels]).item()
#                 test_mnsd += seg_metrics.calculate_mean_nsd([preds], [labels], distance_maps).item()
#                 test_seg_score += seg_metrics.calculate_segmentation_score(test_miou, test_mnsd)
#
#         self.record_metrics(test_loss, test_iou, test_dice, len(test_loader), test_miou, test_mnsd, test_seg_score, train=False)
#         self.scheduler.step(test_loss / len(test_loader))
#         return {'loss': test_loss / len(test_loader), 'iou': test_iou / len(test_loader), 'dice': test_dice / len(test_loader),
#                 'miou': test_miou / len(test_loader), 'mnsd': test_mnsd / len(test_loader), 'seg_score': test_seg_score / len(test_loader)}
#
#     def record_metrics(self, total_loss, total_iou, total_dice, num_items, total_miou, total_mnsd, total_seg_score, train=True):
#         prefix = 'train' if train else 'test'
#         self.metrics[f'{prefix}_loss'].append(total_loss / num_items)
#         self.metrics[f'{prefix}_iou'].append(total_iou / num_items)
#         self.metrics[f'{prefix}_dice'].append(total_dice / num_items)
#         self.metrics[f'{prefix}_miou'].append(total_miou / num_items)
#         self.metrics[f'{prefix}_mnsd'].append(total_mnsd / num_items)
#         self.metrics[f'{prefix}_seg_score'].append(total_seg_score / num_items)
#         if train:
#             self.metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
#
#     def save_results(self):
#         df = pd.DataFrame(self.metrics)
#         df.to_csv('training_metrics.csv', index=False)
#         torch.save(self.model.state_dict(), 'unet_model.pth')
#         print("Training completed and metrics saved to 'training_metrics.csv'")
#
# if __name__ == "__main__":
#     trainer = ModelTrainer("C:/Users/User/Desktop/Shahbaz_project/SAR_RARP50_challenge/Database/train_data.csv",
#                            "C:/Users/User/Desktop/Shahbaz_project/SAR_RARP50_challenge/Database/test_data.csv")
#
#     trainer.train()


import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from data_loader import create_data_loaders
from monai.metrics import compute_iou, compute_generalized_dice
from model.unet import UNet  # Ensure UNet is defined in 'unet.py'
from callback import EarlyStopping


class ModelTrainer:
    """
    Class to manage the training process of a UNet model for semantic segmentation.

    Parameters:
        train_csv_file (str): File path for the training dataset CSV.
        test_csv_file (str): File path for the testing dataset CSV.
        num_epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training and evaluation.
    """

    def __init__(self, train_csv_file, test_csv_file, num_epochs=1000, batch_size=32):
        self.train_csv_file = train_csv_file
        self.test_csv_file = test_csv_file
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(n_channels=3, n_classes=10)
        self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
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

            with autocast():
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
        preds = torch.argmax(outputs, dim=1)  # Shape: [Batch, H, W]
        masks = masks.squeeze(1)  # Shape: [Batch, H, W]

        if preds.shape != masks.shape:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, masks={masks.shape}")

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
        torch.save(self.model.state_dict(), "../plots/unet_model.pth")
        print("Training completed and metrics saved to 'training_metrics.csv'")


if __name__ == "__main__":
    trainer = ModelTrainer(
        "C:/Users/User/Desktop/Shahbaz_project/SAR_RARP50_challenge/Database/train_data.csv",
        "C:/Users/User/Desktop/Shahbaz_project/SAR_RARP50_challenge/Database/test_data.csv"
    )
    trainer.train()
