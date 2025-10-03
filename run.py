from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io

# Khởi tạo FastAPI
app = FastAPI()

# Load model YOLO đã train
model = YOLO("123_aug90_model_ft/weights/best.pt")

# Hàm căn chỉnh thẻ
def align_card(img_bgr, corner_boxes, output_size=(856, 540)):
    src_points = np.array([
        corner_boxes['top_left'],
        corner_boxes['top_right'],
        corner_boxes['bottom_right'],
        corner_boxes['bottom_left']
    ], dtype=np.float32)

    w, h = output_size
    dst_points = np.array([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    aligned_bgr = cv2.warpPerspective(img_bgr, M, (w, h))
    return aligned_bgr

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Đọc ảnh upload
    file_bytes = await file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # YOLO detect
    results = model(img_bgr)
    boxes = results[0].boxes
    names = model.names

    # Map 4 góc
    corner_boxes = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        corner_boxes[label] = [cx, cy]

    # Kiểm tra đủ 4 góc
    required = {"top_left", "top_right", "bottom_right", "bottom_left"}
    if not required.issubset(corner_boxes.keys()):
        return JSONResponse({"error": "Không detect đủ 4 góc"}, status_code=400)

    # Warp thẻ
    aligned_bgr = align_card(img_bgr, corner_boxes)

    # Crop portrait
    portrait_bgr = aligned_bgr[180:480, 15:220]

    # Encode ảnh ra PNG trong bộ nhớ
    _, buffer = cv2.imencode(".png", portrait_bgr)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
