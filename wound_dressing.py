import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from tkinter import Tk, filedialog

# ---------------- FILE INPUT ----------------
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Select image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not file_path:
    exit()

image = cv2.imread(file_path)
original = image.copy()

# ---------------- YOLO DETECTION ----------------
yolo = YOLO("wound.pt")
results = yolo(image)[0]

if len(results.boxes) == 0:
    print("No wound detected")
    exit()

x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy().astype(int)

# ---------------- SAM SEGMENTATION ----------------
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)

input_box = np.array([x1, y1, x2, y2])
masks, _, _ = predictor.predict(box=input_box, multimask_output=False)

wound_mask = (masks[0].astype(np.uint8)) * 255

# ---------------- CONTOUR ----------------
contours, _ = cv2.findContours(
    wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
largest = max(contours, key=cv2.contourArea)

# ---------------- ARM CURVATURE ESTIMATION ----------------
rect = cv2.minAreaRect(largest)
(center_x, center_y), (w, h), angle = rect

if w < h:
    w, h = h, w

w = int(w * 1.5)
h = int(h * 1.3)

# curvature map (simulated cylindrical arm surface)
curvature = np.linspace(-1, 1, w)
curvature = 1 - (curvature ** 2)
curvature = np.tile(curvature, (h, 1))

# ---------------- LIGHTING ESTIMATION ----------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lighting = cv2.GaussianBlur(gray, (101,101), 0) / 255.0

# ---------------- SKIN COLOR ESTIMATION ----------------
cx, cy = int(center_x), int(center_y)
patch = image[max(0,cy-40):cy+40, max(0,cx-40):cx+40]
avg = patch.mean(axis=(0,1)) if patch.size else np.array([210,180,140])

adhesive_color = (
    int(avg[0]*0.85),
    int(avg[1]*0.85),
    int(avg[2]*0.85)
)

# ---------------- DIFFUSION-STYLE GAUZE TEXTURE ----------------
bandage = np.zeros((h, w, 3), dtype=np.uint8)
alpha = np.zeros((h, w), dtype=np.float32)

cv2.rectangle(bandage, (0,0), (w,h), adhesive_color, -1)
alpha[:] = 0.85

pad_w = int(w * 0.45)
pad_h = int(h * 0.6)
px = (w - pad_w)//2
py = (h - pad_h)//2

gauze = np.full((pad_h, pad_w, 3), 240, dtype=np.uint8)

for _ in range(6):
    noise = np.random.normal(0, 6, (pad_h, pad_w, 3))
    gauze = np.clip(gauze + noise, 200, 255)

bandage[py:py+pad_h, px:px+pad_w] = gauze

# ---------------- SKIN COMPRESSION EFFECT ----------------
edge_mask = np.zeros((h, w), dtype=np.float32)
cv2.rectangle(edge_mask, (0,0), (w,h), 1, -1)
edge_mask = cv2.GaussianBlur(edge_mask, (61,61), 0)

compression = edge_mask * 0.08

for c in range(3):
    bandage[:,:,c] = np.clip(
        bandage[:,:,c] - (compression * 255),
        0,255
    )

# ---------------- APPLY CURVATURE + LIGHTING ----------------
for c in range(3):
    bandage[:,:,c] = bandage[:,:,c] * curvature

alpha = cv2.GaussianBlur(alpha, (41,41), 0)

# ---------------- WARP TO WOUND ----------------
box = cv2.boxPoints(rect)
box = np.int32(box)

src = np.float32([[0,0],[w,0],[w,h],[0,h]])
dst = np.float32(box)

M = cv2.getPerspectiveTransform(src, dst)

warped_bandage = cv2.warpPerspective(
    bandage, M, (image.shape[1], image.shape[0])
)

warped_alpha = cv2.warpPerspective(
    alpha, M, (image.shape[1], image.shape[0])
)

# ---------------- SHADOW + BLEND (SAFE) ----------------
shadow = cv2.GaussianBlur(warped_alpha, (21,21), 0) * 0.5

image_float = image.astype(np.float32)

for c in range(3):
    image_float[:,:,c] *= (1 - shadow)

for c in range(3):
    image_float[:,:,c] = (
        warped_alpha * warped_bandage[:,:,c] +
        (1 - warped_alpha) * image_float[:,:,c]
    )

image = np.clip(image_float, 0, 255).astype(np.uint8)


# ---------------- DISPLAY ----------------
cv2.imshow("Original", cv2.resize(original,(500,400)))
cv2.imshow("Ultra Medical Dressing", cv2.resize(image,(500,400)))
cv2.waitKey(0)
cv2.destroyAllWindows()

