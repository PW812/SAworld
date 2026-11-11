"""Pose/time metadata for the streaming SAWorld residual memory."""
import torch
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class SAWorldMemoryMetadata:
    """Use the same key-camera ego frame as BEVDepth4D.prepare_inputs.

    Image resize/flip is already accounted for by the camera projection. BDA
    transforms BEV coordinates, so its inverse belongs in BEV-to-global.
    """

    def __call__(self, results):
        inputs = results['img_inputs']
        key_ego_to_global = inputs[2][0].double()
        bev_augmentation = torch.eye(4, dtype=torch.float64)
        bev_augmentation[:3, :3] = inputs[6].double()
        results['memory_ego2global'] = (
            key_ego_to_global @ torch.linalg.inv(bev_augmentation)).numpy()
        results['memory_timestamp'] = float(results['timestamp'])
        return results
