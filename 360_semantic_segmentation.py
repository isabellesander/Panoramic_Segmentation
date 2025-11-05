from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import py360convert
import torch

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# Device selection: prefer MPS (Apple silicon) then CUDA, else CPU
import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print('Using device: MPS (Apple GPU)')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f'Using device: CUDA (GPU) - {torch.cuda.get_device_name(0)}')
else:
    DEVICE = torch.device('cpu')
    print('Using device: CPU')

image_name = "nng"  # Change to your 360° panorama filename


# ---------------------------
# 1. Load ADE20K Mask2Former Model
# ---------------------------
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
# Load model and move to device
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-small-ade-semantic",
)
model.to(DEVICE)
model.eval()

# ---------------------------
# 2. ADE20K Color Palette
# ---------------------------
ADE20k_COLORS = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230], [4, 250, 7],
    [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255],
    [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140], [250, 10, 15],
    [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255 ,82, 0], [0, 255, 245], [0, 61, 255], [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0],
    [0, 143, 255], [51, 255, 0], [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255], [255, 0, 204], [0, 255, 194],
    [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255], [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0], [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255],
    [0, 214, 255], [255, 0, 112], [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0], [10, 190, 212], [214, 255, 0],
    [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204], [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255], [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]
])

# You can set this to an int class id to FORCE the bottom face, or to None to segment normally.
OVERRIDE_BOTTOM_CLASS_ID = 11   # e.g. 6 for road, or None

# (Kept for backward compatibility if you still reference SIDEWALK_ID elsewhere)
SIDEWALK_ID = OVERRIDE_BOTTOM_CLASS_ID

# Apply ADE color palette
def apply_color_palette(segmentation_map, palette):
    colored_image = np.zeros((*segmentation_map.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        colored_image[segmentation_map == class_idx] = color
    return colored_image

# Segmentation function: outputs class index map
def segment_image(pil_img):
    inputs = image_processor(pil_img, return_tensors="pt")
    # Move tensors to the selected device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Post-process outputs (returns a tensor on the model device). Move to CPU before converting to numpy.
    seg_map = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(pil_img.height, pil_img.width)]
    )[0]
    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.to('cpu').numpy()
    return seg_map

# ---------------------------
# 3. Load panorama and create cube faces
# ---------------------------
equirect_img = Image.open(image_name+".jpg").convert("RGB")
equirect_np = np.array(equirect_img)

# Convert to 6 cube faces (front, right, back, left, top, bottom)
cube_faces = py360convert.e2c(equirect_np, face_w=512, mode='bilinear', cube_format='list')

# ---------------------------
# 4. Process each face, force bottom to sidewalk
# ---------------------------
segmented_faces_rgb = []
for i, face in enumerate(cube_faces):
    if i == 5 and OVERRIDE_BOTTOM_CLASS_ID is not None:
        # Force entire bottom face to the chosen class id
        seg_map = np.full((face.shape[0], face.shape[1]), OVERRIDE_BOTTOM_CLASS_ID, dtype=np.int32)
    else:
        # Normal segmentation
        seg_map = segment_image(Image.fromarray(face))
    seg_rgb = apply_color_palette(seg_map, ADE20k_COLORS)
    segmented_faces_rgb.append(seg_rgb)

# ---------------------------
# 5. Merge back into equirectangular
# ---------------------------
segmented_equirect = py360convert.c2e(
    segmented_faces_rgb,
    h=equirect_np.shape[0],
    w=equirect_np.shape[1],
    mode='bilinear',
    cube_format='list'
)

# ---------------------------
# 6. Save and show results
# ---------------------------
seg_result_img = Image.fromarray(segmented_equirect)
seg_result_img.save(image_name+"_360_segmented.png")

# plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(equirect_img)
# plt.title("Original 360° Panorama")
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(seg_result_img)
# plt.title("Segmented Panorama (Bottom = Sidewalk)")
# plt.axis("off")
# plt.show()