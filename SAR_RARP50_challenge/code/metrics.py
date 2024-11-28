import torch
from scipy.ndimage import distance_transform_edt


class SegmentationMetrics:
    """
    A class for computing segmentation metrics, including IoU, Dice score, and NSD (Normalized Surface Distance).
    """

    def __init__(self, threshold=1.5):
        """
        Initialize the SegmentationMetrics object.

        Args:
            threshold (float): Threshold for NSD computation.
        """
        self.threshold = threshold

    def calculate_iou(self, preds, labels):
        """
        Calculate Intersection over Union (IoU) for predictions and labels.

        Args:
            preds (torch.Tensor): Predicted binary mask.
            labels (torch.Tensor): Ground truth binary mask.

        Returns:
            torch.Tensor: Mean IoU across the batch.
        """
        intersection = (preds & labels).float().sum((1, 2))
        union = (preds | labels).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    def calculate_dice(self, preds, labels):
        """
        Calculate the Dice coefficient for predictions and labels.

        Args:
            preds (torch.Tensor): Predicted binary mask.
            labels (torch.Tensor): Ground truth binary mask.

        Returns:
            torch.Tensor: Mean Dice score across the batch.
        """
        intersection = (preds & labels).float().sum((1, 2))
        dice = (2. * intersection + 1e-6) / (preds.sum((1, 2)) + labels.sum((1, 2)) + 1e-6)
        return dice.mean()

    def convert_to_boolean_tensors(self, outputs, masks):
        """
        Convert model outputs and ground truth masks to binary tensors.

        Args:
            outputs (torch.Tensor): Model's raw outputs.
            masks (torch.Tensor): Ground truth masks.

        Returns:
            tuple: Binary tensors for predictions and labels.
        """
        preds = torch.sigmoid(outputs) > 0.5
        labels = masks > 0.5
        return preds, labels

    def calculate_nsd(self, preds, labels, distance_map):
        """
        Calculate the Normalized Surface Distance (NSD).

        Args:
            preds (torch.Tensor): Predicted binary mask.
            labels (torch.Tensor): Ground truth binary mask.
            distance_map (torch.Tensor): Distance map computed from ground truth mask.

        Returns:
            float: NSD score.
        """
        if preds.sum() == 0 or labels.sum() == 0:
            print("Warning: Empty predictions or labels detected during NSD computation.")
            return 0.0

        pred_within_threshold = (distance_map * preds) <= self.threshold
        target_within_threshold = (distance_map * labels) <= self.threshold
        num_pred_within_threshold = pred_within_threshold.sum().item()
        num_target_within_threshold = target_within_threshold.sum().item()
        total_pred = preds.sum().item()
        total_target = labels.sum().item()

        if total_pred + total_target == 0:
            print("Warning: No boundary pixels found in predictions or labels.")
            return 0.0

        nsd = (num_pred_within_threshold + num_target_within_threshold + 1e-6) / (total_pred + total_target + 1e-6)
        return nsd

    def calculate_mean_iou(self, predictions, masks):
        """
        Calculate the mean IoU across all samples.

        Args:
            predictions (list[torch.Tensor]): List of predicted binary masks.
            masks (list[torch.Tensor]): List of ground truth binary masks.

        Returns:
            torch.Tensor: Mean IoU across all samples.
        """
        ious = [self.calculate_iou(pred, mask) for pred, mask in zip(predictions, masks)]
        return torch.tensor(ious).mean()

    def calculate_mean_nsd(self, predictions, masks, distance_maps):
        """
        Calculate the mean NSD across all samples.

        Args:
            predictions (list[torch.Tensor]): List of predicted binary masks.
            masks (list[torch.Tensor]): List of ground truth binary masks.
            distance_maps (list[torch.Tensor]): List of distance maps for ground truth masks.

        Returns:
            torch.Tensor: Mean NSD across all samples.
        """
        nsds = [self.calculate_nsd(pred, mask, dist_map) for pred, mask, dist_map in zip(predictions, masks, distance_maps)]
        return torch.tensor(nsds).mean()

    def calculate_segmentation_score(self, mean_iou, mean_nsd):
        """
        Calculate a combined segmentation score using mean IoU and mean NSD.

        Args:
            mean_iou (float or torch.Tensor): Mean IoU score.
            mean_nsd (float or torch.Tensor): Mean NSD score.

        Returns:
            float: Combined segmentation score.
        """
        mean_iou = torch.tensor(mean_iou) if not isinstance(mean_iou, torch.Tensor) else mean_iou
        mean_nsd = torch.tensor(mean_nsd) if not isinstance(mean_nsd, torch.Tensor) else mean_nsd

        if torch.isnan(mean_iou) or torch.isnan(mean_nsd):
            print(f"Warning: NaN encountered in segmentation score calculation. mIoU: {mean_iou}, mNSD: {mean_nsd}")
            return float('nan')

        return (mean_iou * mean_nsd).sqrt().item()

    def compute_distance_map(self, mask):
        """
        Compute the distance map from the ground truth mask.

        Args:
            mask (torch.Tensor): Ground truth binary mask.

        Returns:
            torch.Tensor: Distance map for the mask.
        """
        if mask.sum() == 0:
            print("Warning: Empty mask detected during distance map computation.")
            return torch.zeros_like(mask, dtype=torch.float32)

        mask_np = mask.cpu().numpy()
        distance_map = distance_transform_edt(mask_np == 0)
        return torch.from_numpy(distance_map).to(mask.device)


# Example usage
if __name__ == "__main__":
    seg_metrics = SegmentationMetrics(threshold=1.5)
    outputs = torch.randn(2, 1, 128, 128)
    masks = torch.tensor([[[0, 1, 1], [0, 1, 1], [0, 0, 0]], [[1, 0, 0], [1, 0, 1], [1, 0, 0]]], dtype=torch.float32)

    preds, labels = seg_metrics.convert_to_boolean_tensors(outputs, masks)
    distance_maps = [seg_metrics.compute_distance_map(label) for label in labels]

    mIoU = seg_metrics.calculate_mean_iou([preds], [labels])
    mNSD = seg_metrics.calculate_mean_nsd([preds], [labels], distance_maps)
    segmentation_score = seg_metrics.calculate_segmentation_score(mIoU, mNSD)

    print(f"Mean IoU: {mIoU:.4f}")
    print(f"Mean NSD: {mNSD:.4f}")
    print(f"Segmentation Score: {segmentation_score:.4f}")

#
# import torch
# from scipy.ndimage import distance_transform_edt
# import torch.nn.functional as F
#
# class SegmentationMetrics:
#     """
#     A class for computing segmentation metrics, including IoU, Dice score, and NSD (Normalized Surface Distance).
#     """
#
#     def __init__(self, threshold=1.5):
#         """
#         Initialize the SegmentationMetrics object with a threshold for NSD computation.
#         Args:
#             threshold (float): Threshold for NSD computation.
#         """
#         self.threshold = threshold
#
#     def extract_boundaries(self, masks):
#         """
#         Extract boundaries from binary segmentation masks using erosion and the original mask.
#         Args:
#             masks (torch.Tensor): Binary segmentation masks (N, 1, H, W).
#         Returns:
#             torch.Tensor: Boundary mask of the same size as input masks.
#         """
#         kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=masks.device)
#         eroded = F.conv2d(masks, kernel, padding=1) >= (kernel.sum() - 1)
#         boundaries = masks.float() - eroded.float()
#         return boundaries
#
#     def calculate_iou(self, preds, labels):
#         """
#         Calculate Intersection over Union (IoU) for predictions and labels.
#         Args:
#             preds (torch.Tensor): Predicted binary mask.
#             labels (torch.Tensor): Ground truth binary mask.
#         Returns:
#             torch.Tensor: Mean IoU across the batch.
#         """
#         intersection = (preds & labels).float().sum((1, 2))
#         union = (preds | labels).float().sum((1, 2))
#         iou = (intersection + 1e-6) / (union + 1e-6)
#         return iou.mean()
#
#     def calculate_dice(self, preds, labels):
#         """
#         Calculate the Dice coefficient for predictions and labels.
#         Args:
#             preds (torch.Tensor): Predicted binary mask.
#             labels (torch.Tensor): Ground truth binary mask.
#         Returns:
#             torch.Tensor: Mean Dice score across the batch.
#         """
#         intersection = (preds & labels).float().sum((1, 2))
#         dice = (2. * intersection + 1e-6) / (preds.sum((1, 2)) + labels.sum((1, 2)) + 1e-6)
#         return dice.mean()
#
#     def convert_to_boolean_tensors(self, outputs, masks):
#         """
#         Convert model outputs and ground truth masks to binary tensors.
#         Args:
#             outputs (torch.Tensor): Model's raw outputs.
#             masks (torch.Tensor): Ground truth masks.
#         Returns:
#             tuple: Binary tensors for predictions and labels.
#         """
#         preds = torch.sigmoid(outputs) > 0.5
#         labels = masks > 0.5
#         return preds, labels
#
#     def calculate_nsd(self, preds, labels):
#         """
#         Calculate NSD using actual boundary distances.
#         Args:
#             preds (torch.Tensor): Predicted binary mask.
#             labels (torch.Tensor): Ground truth binary mask.
#         Returns:
#             float: NSD score.
#         """
#         preds_boundary = self.extract_boundaries(preds)
#         labels_boundary = self.extract_boundaries(labels)
#
#         distance_map_pred = distance_transform_edt(1 - labels_boundary.squeeze(1).cpu().numpy())
#         distance_map_label = distance_transform_edt(1 - preds_boundary.squeeze(1).cpu().numpy())
#
#         pred_surface_distances = torch.from_numpy(distance_map_pred).to(preds.device)[preds_boundary.squeeze(1).bool()].sum()
#         label_surface_distances = torch.from_numpy(distance_map_label).to(labels.device)[labels_boundary.squeeze(1).bool()].sum()
#
#         total_distances = pred_surface_distances + label_surface_distances
#         total_boundaries = preds_boundary.sum() + labels_boundary.sum()
#
#         nsd = (total_distances + 1e-6) / (total_boundaries + 1e-6)
#         return nsd
#
#     def calculate_mean_iou(self, predictions, masks):
#         """
#         Calculate the mean IoU across all samples.
#         Args:
#             predictions (list[torch.Tensor]): List of predicted binary masks.
#             masks (list[torch.Tensor]): List of ground truth binary masks.
#         Returns:
#             torch.Tensor: Mean IoU across all samples.
#         """
#         ious = [self.calculate_iou(pred, mask) for pred, mask in zip(predictions, masks)]
#         return torch.tensor(ious).mean()
#
#     def calculate_mean_nsd(self, predictions, masks):
#         """
#         Calculate the mean NSD across all samples.
#         Args:
#             predictions (list[torch.Tensor]): List of predicted binary masks.
#             masks (list[torch.Tensor]): List of ground truth binary masks.
#         Returns:
#             torch.Tensor: Mean NSD across all samples.
#         """
#         nsds = [self.calculate_nsd(pred, mask) for pred, mask in zip(predictions, masks)]
#         return torch.tensor(nsds).mean()
#
#     def calculate_segmentation_score(self, mean_iou, mean_nsd):
#         """
#         Calculate a combined segmentation score using mean IoU and mean NSD.
#         Args:
#             mean_iou (float or torch.Tensor): Mean IoU score.
#             mean_nsd (float or torch.Tensor): Mean NSD score.
#         Returns:
#             float: Combined segmentation score.
#         """
#         mean_iou = torch.tensor(mean_iou) if not isinstance(mean_iou, torch.Tensor) else mean_iou
#         mean_nsd = torch.tensor(mean_nsd) if not isinstance(mean_nsd, torch.Tensor) else mean_nsd
#         if torch.isnan(mean_iou) or torch.isnan(mean_nsd):
#             print(f"Warning: NaN encountered in segmentation score calculation. mIoU: {mean_iou}, mNSD: {mean_nsd}")
#             return float('nan')
#         return (mean_iou * mean_nsd).sqrt().item()
#
#     def compute_distance_map(self, mask):
#         """
#         Compute the distance map from the ground truth mask.
#         Args:
#             mask (torch.Tensor): Ground truth binary mask.
#         Returns:
#             torch.Tensor: Distance map for the mask.
#         """
#         if mask.sum() == 0:
#             print("Warning: Empty mask detected during distance map computation.")
#             return torch.zeros_like(mask, dtype=torch.float32)
#         mask_np = mask.squeeze(1).cpu().numpy()  # Make sure to handle dimensions properly
#         distance_map = distance_transform_edt(mask_np == 0)
#         return torch.from_numpy(distance_map).to(mask.device)
#
#
# if __name__ == "__main__":
#     seg_metrics = SegmentationMetrics(threshold=1.5)
#     outputs = torch.randn(2, 1, 3, 3)  # Example outputs from a model
#     masks = torch.tensor([[[0, 1, 1], [0, 1, 1], [0, 0, 0]], [[1, 0, 0], [1, 0, 1], [1, 0, 0]]], dtype=torch.float32).unsqueeze(1)
#
#     preds, labels = seg_metrics.convert_to_boolean_tensors(outputs, masks)
#     mIoU = seg_metrics.calculate_mean_iou([preds], [labels])
#     mNSD = seg_metrics.calculate_mean_nsd([preds], [labels])
#
#     print(f"Mean IoU: {mIoU:.4f}")
#     print(f"Mean NSD: {mNSD:.4f}")
