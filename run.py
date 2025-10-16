from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from typing import List
import cv2
import numpy as np
import io
import uuid
import base64

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
    jpeg_quality = 80
    _, buffer = cv2.imencode(".jpg", portrait_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


# --- Endpoint mới xử lý batch ảnh ---
@app.post("/detect_batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Nhận một batch ảnh, xử lý và trả về kết quả chi tiết cho từng ảnh,
    bao gồm ảnh chân dung dưới dạng base64 khi thành công hoặc lỗi có cấu trúc khi thất bại.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Không có tệp nào được tải lên.")

    # Thêm ID duy nhất cho mỗi request, rất hữu ích cho việc log và debug
    request_id = str(uuid.uuid4())

    batch_results = []
    success_count = 0
    failed_count = 0

    for file in files:
        file_name = file.filename

        # Cấu trúc response nhất quán cho mỗi file
        result_item = {
            "file_name": file_name,
            "status": None,
            "data": None,
            "error": None
        }

        try:
            # 1. Đọc và giải mã ảnh
            file_bytes = await file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError("Không thể giải mã tệp thành ảnh hợp lệ.")

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
                # Ném ra lỗi cụ thể để khối except có thể bắt được
                raise ValueError(f"Không detect đủ 4 góc. Các góc tìm thấy: {list(corner_boxes.keys())}")

            # 5. Warp thẻ và Crop chân dung
            aligned_bgr = align_card(img_bgr, corner_boxes)
            portrait_bgr = aligned_bgr[180:480, 15:220]

            # 6. Mã hóa ảnh chân dung thành chuỗi base64
            _, buffer = cv2.imencode(".png", portrait_bgr)
            portrait_base64 = base64.b64encode(buffer).decode("utf-8")

            # 7. Ghi nhận kết quả thành công
            success_count += 1
            result_item["status"] = "success"
            result_item["data"] = {
                "portrait_image_base64": portrait_base64
            }

        except Exception as e:
            # 8. Ghi nhận kết quả thất bại với lỗi có cấu trúc
            failed_count += 1
            result_item["status"] = "failed"
            result_item["error"] = {
                "code": "PROCESSING_ERROR",  # Mã lỗi để client dễ xử lý
                "message": str(e)
            }

        batch_results.append(result_item)

    # Cấu trúc JSON response cuối cùng, có thêm phần tóm tắt
    final_response = {
        "request_id": request_id,
        "summary": {
            "total_files": len(files),
            "success_count": success_count,
            "failed_count": failed_count
        },
        "results": batch_results
    }

    return JSONResponse(final_response)