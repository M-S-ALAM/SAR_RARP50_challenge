import numpy as np
import torch
from monai.metrics import compute_generalized_dice, compute_iou
from monai.transforms import AsDiscrete


def calculate_metrics(predictions, ground_truth, include_nsd=False):
    """
    Calculate mIoU_avg, NSD_avg, and final segmentation score.

    Parameters:
    - predictions (np.ndarray): Predicted segmentation maps (B, H, W or B, D, H, W).
    - ground_truth (np.ndarray): Ground truth segmentation maps (same shape as predictions).
    - include_nsd (bool): Whether to calculate NSD (Normalized Surface Distance).

    Returns:
    - mIoU_avg: Average mIoU across samples.
    - NSD_avg: Average NSD across samples (if include_nsd=True).
    - final_score: Product of mIoU_avg and NSD_avg.
    """
    # Convert to PyTorch tensors
    predictions = torch.tensor(predictions, dtype=torch.float32)
    ground_truth = torch.tensor(ground_truth, dtype=torch.float32)

    # Ensure predictions and ground truth are binary
    pred_discrete = AsDiscrete(to_onehot=None)(predictions)
    gt_discrete = AsDiscrete(to_onehot=None)(ground_truth)

    # Compute mIoU for each sample
    miou_scores = compute_iou(pred_discrete, gt_discrete, include_background=False)
    mIoU_avg = miou_scores.mean().item()

    # Compute NSD if requested
    if include_nsd:
        nsd_scores = []
        for pred, gt in zip(predictions, ground_truth):
            nsd = compute_generalized_dice(pred.unsqueeze(0), gt.unsqueeze(0))
            # Aggregate NSD values (e.g., mean across all classes/regions)
            nsd_scores.append(nsd.mean().item())
        NSD_avg = np.mean(nsd_scores)
    else:
        NSD_avg = 1  # Default to 1 if NSD is not used

    # Calculate final segmentation score
    final_score = mIoU_avg * NSD_avg

    return mIoU_avg, NSD_avg, final_score


# Example usage
if __name__ == "__main__":
    # Dummy data for predictions and ground truth
    predictions = np.random.randint(0, 2, (10, 128, 128))  # 10 binary segmentation maps
    ground_truth = np.random.randint(0, 2, (10, 128, 128))  # 10 binary ground truth maps

    # Calculate metrics
    mIoU_avg, NSD_avg, final_score = calculate_metrics(predictions, ground_truth, include_nsd=True)

    print(f"mIoU_avg: {mIoU_avg}")
    print(f"NSD_avg: {NSD_avg}")
    print(f"Final Score: {final_score}")