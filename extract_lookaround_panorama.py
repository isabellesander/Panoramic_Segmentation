from io import BytesIO
from PIL import Image
from pillow_heif import register_heif_opener
from streetlevel import lookaround

register_heif_opener()  # allow PIL to open HEIC/HEIF

tile = lookaround.get_coverage_tile_by_latlon(52.502898, 13.3733425)  # CoverageTile
panos = tile.panos  # list[LookaroundPanorama]

if not panos:
    raise RuntimeError("No panoramas found on this tile.")

# pick by ID if present; else fall back to first pano
target_id = 17229882189354874571



pano = next((p for p in panos if getattr(p, "id", None) == target_id), panos[0])
if getattr(pano, "id", None) != target_id:
    print(f"Requested ID not found. Using first pano: {getattr(pano, 'id', None)}")

auth = lookaround.Authenticator()
zoom = 2

# Download and decode 6 HEIC faces -> PIL Images
faces = []
for face_idx in range(6):
    face_heic = lookaround.get_panorama_face(pano, face_idx, zoom, auth)  # bytes
    img = Image.open(BytesIO(face_heic)).convert("RGB")
    faces.append(img)

# Optional: stitch to equirectangular (requires torch)
try:
    import torch  # noqa: F401
except ImportError:
    raise SystemExit("PyTorch not installed. Install with: pip install torch")

eq = lookaround.to_equirectangular(faces, pano.camera_metadata)
eq.save(f"{getattr(pano, 'id', 'pano')}_{zoom}.jpg", quality=100)
print("Saved panorama.")