import torch
from scipy.ndimage import distance_transform_edt

class SegmentationMetrics:
    def __init__(self, threshold=1.5):
        self.threshold = threshold

    def calculate_iou(self, preds, labels):
        intersection = (preds & labels).float().sum((1, 2))
        union = (preds | labels).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()

    def calculate_dice(self, preds, labels):
        intersection = (preds & labels).float().sum((1, 2))
        dice = (2. * intersection + 1e-6) / (preds.sum((1, 2)) + labels.sum((1, 2)) + 1e-6)
        return dice.mean()

    def convert_to_boolean_tensors(self, outputs, masks):
        preds = torch.sigmoid(outputs) > 0.5
        labels = masks > 0.5
        return preds, labels

    def calculate_nsd(self, preds, labels, distance_map):
        if preds.sum() == 0 or labels.sum() == 0:
            print("Warning: Empty predictions or labels detected during NSD computation.")
            return 0.0  # No boundaries exist

        if torch.isnan(distance_map).any() or torch.isinf(distance_map).any():
            print("Error: Invalid distance map detected during NSD computation.")
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
        ious = [self.calculate_iou(pred, mask) for pred, mask in zip(predictions, masks)]
        return torch.tensor(ious).mean()

    def calculate_mean_nsd(self, predictions, masks, distance_maps):
        nsds = [self.calculate_nsd(pred, mask, dist_map) for pred, mask, dist_map in zip(predictions, masks, distance_maps)]
        return torch.tensor(nsds).mean()

    def calculate_segmentation_score(self, mean_iou, mean_nsd):
        mean_iou = torch.tensor(mean_iou) if not isinstance(mean_iou, torch.Tensor) else mean_iou
        mean_nsd = torch.tensor(mean_nsd) if not isinstance(mean_nsd, torch.Tensor) else mean_nsd
        if torch.isnan(mean_iou) or torch.isnan(mean_nsd):
            print(f"Warning: NaN encountered in segmentation score calculation. mIoU: {mean_iou}, mNSD: {mean_nsd}")
            return float('nan')
        return (mean_iou * mean_nsd).sqrt().item()

    def compute_distance_map(self, mask):
        if mask.sum() == 0:
            # Log a warning or handle empty mask case appropriately
            print("Warning: Empty mask detected during distance map computation.")
            # Return a distance map of zeros with the same shape as the mask
            return torch.zeros_like(mask, dtype=torch.float32)
        else:
            mask_np = mask.cpu().numpy()  # Convert to numpy
            distance_map = distance_transform_edt(mask_np == 0)  # Compute distance to the nearest background
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
