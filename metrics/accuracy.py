import torch

def voxel_accuracy(
    voxel_true:torch.Tensor,
    voxel_pred:torch.Tensor,
) -> float:
    assert voxel_true.ndim == 5 # (b, c, x, y, z)
    assert voxel_true.shape == voxel_pred.shape
    assert voxel_true.dtype == torch.float32
    assert voxel_true.dtype == voxel_pred.dtype

    voxel_pred = voxel_pred >= 0.5
    voxel_pred = voxel_pred.to(torch.float32)
    accuracy = voxel_true == voxel_pred
    accuracy = accuracy.to(torch.float32)
    accuracy = torch.mean(accuracy)
    return accuracy.item()