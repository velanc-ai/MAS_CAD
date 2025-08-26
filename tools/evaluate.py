import trimesh
import numpy as np

def voxelize_mesh(mesh, pitch=1.0):
    # Convert mesh into a voxel grid
    voxel = mesh.voxelized(pitch=pitch)
    return voxel.matrix

def compute_iou(mesh1, mesh2, pitch=1.0):
    vox1 = voxelize_mesh(mesh1, pitch)
    vox2 = voxelize_mesh(mesh2, pitch)

    # Make sure same grid size
    min_shape = np.minimum(vox1.shape, vox2.shape)
    vox1 = vox1[:min_shape[0], :min_shape[1], :min_shape[2]]
    vox2 = vox2[:min_shape[0], :min_shape[1], :min_shape[2]]

    intersection = np.logical_and(vox1, vox2).sum()
    union = np.logical_or(vox1, vox2).sum()

    return intersection / union if union > 0 else 0.0

# Example usage:
mesh_gt = trimesh.load("ground_truth.stl")
mesh_pred = trimesh.load("generated.stl")

iou_score = compute_iou(mesh_gt, mesh_pred, pitch=0.5)
print("3D IoU:", iou_score)
