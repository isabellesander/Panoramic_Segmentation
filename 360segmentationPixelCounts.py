import numpy as np
from PIL import Image
import py360convert
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import csv

image_name = "gleisdreieck2"  # Change to your 360° panorama filename

# Panorama input (change if needed)
PANORAMA_PATH = image_name+".jpg"

# Cube face resolution (lower => faster)
FACE_RES = 512  # try 256 for speed

# Optional: force one face to a fixed class (kept from your previous logic)
FORCE_FACE_INDEX = 5          # set to None to disable
FORCE_CLASS_ID = 11        # e.g. 11 = sidewalk in ADE20K

# Load model 
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic").eval()

# (Optional GPU)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using device: MPS (Apple GPU)')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using device: CUDA (GPU) - {torch.cuda.get_device_name(0)}')
else:
    device = torch.device('cpu')
    print('Using device: CPU')
model.to(device)

def segment_face(face_np):
    pil_face = Image.fromarray(face_np)
    inputs = image_processor(pil_face, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    seg = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(pil_face.height, pil_face.width)]
    )[0]
    return seg.cpu().numpy().astype(np.int32)

# Solid-angle weights for cubemap pixels (vectorized, per face)
def cubeface_uv(N):
    # pixel-center coords in [-1, 1]
    xs = (np.arange(N) + 0.5) / N * 2.0 - 1.0  # shape (N,)
    ys = (np.arange(N) + 0.5) / N * 2.0 - 1.0
    u, v = np.meshgrid(xs, ys)  # u->cols (x), v->rows (y)
    return u, v

def cubeface_solid_angle_weights(N):
    # dOmega ≈ (du*dv) / (1 + u^2 + v^2)^(3/2), with du=dv=2/N
    u, v = cubeface_uv(N)
    du = dv = 2.0 / N
    w = (du * dv) / np.power(1.0 + u*u + v*v, 1.5)  # shape (N,N)
    return w


def main():
    # Load panorama
    pano_img = Image.open(PANORAMA_PATH).convert("RGB")
    pano_np = np.array(pano_img)

    # Split into cube faces
    cube_faces = py360convert.e2c(
        pano_np,
        face_w=FACE_RES,
        mode="bilinear",
        cube_format="list"
    )

    # Segment each face (keep indices only)
    seg_faces_indices = []
    for i, face in enumerate(cube_faces):
        if FORCE_FACE_INDEX is not None and i == FORCE_FACE_INDEX:
            seg_map = np.full(face.shape[:2], FORCE_CLASS_ID, dtype=np.int32)
        else:
            seg_map = segment_face(face)
        seg_faces_indices.append(seg_map)

    # Count pixels per class (raw and solid-angle weighted)
    N = seg_faces_indices[0].shape[0]
    w_face = cubeface_solid_angle_weights(N)  # shape (N,N), same for all faces

    num_classes = int(max(f.max() for f in seg_faces_indices)) + 1
    counts_raw = np.zeros(num_classes, dtype=np.int64)
    counts_ang = np.zeros(num_classes, dtype=np.float64)

    for f in range(6):
        labels = seg_faces_indices[f].ravel()
        w = w_face.ravel()
        counts_raw += np.bincount(labels, minlength=num_classes)
        counts_ang += np.bincount(labels, weights=w, minlength=num_classes)

    total_pixels = 6 * N * N
    total_solid_angle = counts_ang.sum()  # ≈ 4π

    present_classes = np.flatnonzero(counts_raw)

    # Console preview
    print("Class counts (raw pixels and solid-angle weighted):")
    for cls in present_classes:
        pct_raw = counts_raw[cls] / total_pixels * 100.0
        pct_ang = counts_ang[cls] / total_solid_angle * 100.0 if total_solid_angle > 0 else 0.0
        print(f"{cls:3d}\traw={counts_raw[cls]:8d} ({pct_raw:6.2f}%)  ang={counts_ang[cls]:10.6f} ({pct_ang:6.2f}%)")

    # Save counts CSV (columns used by SegmentationStats)
    with open(image_name+"_panorama_class_counts.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # Keep both raw and angular; provide 'angular_weighted_count' and 'percent' for downstream
        writer.writerow(["class_id",
                         "pixel_count_raw", "percent_raw",
                         "angular_weighted_count", "percent"])
        for cls in present_classes:
            pct_raw = counts_raw[cls] / total_pixels if total_pixels > 0 else 0.0
            pct_ang = counts_ang[cls] / total_solid_angle if total_solid_angle > 0 else 0.0
            writer.writerow([cls,
                             int(counts_raw[cls]), f"{pct_raw*100:.4f}",
                             f"{counts_ang[cls]:.6f}", f"{pct_ang*100:.4f}"])

    print("Saved:", image_name+"_panorama_class_counts.csv")

    # (Optional) If you still want an equirect label map for visualization, keep this:
    # seg_faces_indices_rgb = [np.repeat(f[..., None], 3, axis=-1).astype(np.uint8)
    #                          for f in seg_faces_indices]
    # label_equirect_rgb = py360convert.c2e(
    #     seg_faces_indices_rgb, h=pano_np.shape[0], w=pano_np.shape[1],
    #     mode="nearest", cube_format="list"
    # )
    # label_map = label_equirect_rgb[..., 0].astype(np.int32)
    # np.save(image_name+"_label_map.npy", label_map)
    # print("Saved:", image_name+"_label_map.npy")

if __name__ == "__main__":
    main()