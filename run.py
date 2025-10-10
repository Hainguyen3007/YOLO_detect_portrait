from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from typing import List
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


# --- Endpoint mới xử lý batch ảnh ---
@app.post("/detect_batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Nhận một batch (danh sách) ảnh, xử lý từng ảnh để detect và căn chỉnh thẻ,
    sau đó cắt crop ảnh chân dung.
    Kết quả trả về là JSON chứa trạng thái xử lý cho từng ảnh.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Không có tệp nào được tải lên.")

    batch_results = []

    for file in files:
        file_name = file.filename
        try:
            # 1. Đọc và giải mã ảnh
            file_bytes = await file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Không thể giải mã tệp thành ảnh.")

            # 2. YOLO detect
            results = model(img_bgr)
            boxes = results[0].boxes
            names = model.names

            # 3. Map 4 góc
            corner_boxes = {}
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                corner_boxes[label] = [cx, cy]

            # 4. Kiểm tra đủ 4 góc
            required = {"top_left", "top_right", "bottom_right", "bottom_left"}
            if not required.issubset(corner_boxes.keys()):
                batch_results.append({
                    "file_name": file_name,
                    "status": "failed",
                    "reason": "Không detect đủ 4 góc"
                })
                continue  # Chuyển sang ảnh tiếp theo

            # 5. Warp thẻ và Crop chân dung
            aligned_bgr = align_card(img_bgr, corner_boxes)
            portrait_bgr = aligned_bgr[180:480, 15:220]

            # NOTE: Để đơn giản hóa output batch, ta chỉ trả về trạng thái thành công.
            # Trong thực tế, bạn có thể cần lưu ảnh đã crop ra một thư mục,
            # hoặc mã hóa nó thành base64 để trả về trong JSON (cần cẩn thận với kích thước response).

            batch_results.append({
                "file_name": file_name,
                "status": "success",
                "message": "Xử lý thành công. Ảnh chân dung đã được crop."
            })

        except Exception as e:
            batch_results.append({
                "file_name": file_name,
                "status": "failed",
                "reason": f"Lỗi xử lý: {str(e)}"
            })

    return JSONResponse(batch_results)