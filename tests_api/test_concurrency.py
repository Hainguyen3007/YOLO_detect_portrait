import requests
import time
import concurrent.futures
import os

API_URL = "http://127.0.0.1:8000/detect"  # đổi nếu API chạy cổng khác
DATASET_FOLDER = "test_tps"  # thư mục chứa ảnh test
CONCURRENCY = 50  # số lượng request chạy song song

def send_request(img_path):
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/jpeg")}
        try:
            resp = requests.post(API_URL, files=files, timeout=60)
            if resp.status_code == 200:
                return True
        except Exception as e:
            print(f"❌ Lỗi với {img_path}: {e}")
    return False

def benchmark_api(folder, concurrency=CONCURRENCY):
    image_files = [os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("❌ Không tìm thấy ảnh trong dataset.")
        return

    print(f"🔄 Bắt đầu benchmark {len(image_files)} ảnh với {concurrency} request song song...")

    start = time.time()

    # Pool để gửi song song
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_img = {executor.submit(send_request, img): img for img in image_files}
        for future in concurrent.futures.as_completed(future_to_img):
            img = future_to_img[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"❌ Lỗi không mong muốn {img}: {exc}")

    end = time.time()

    total_time = end - start
    success_count = sum(1 for r in results if r)

    if total_time > 0:
        tps = success_count / total_time
        print(f"✅ Thành công {success_count}/{len(image_files)} ảnh")
        print(f"⏱️ Tổng thời gian: {total_time:.2f} giây")
        print(f"⚡ TPS (concurrent {concurrency}): {tps:.2f} ảnh/giây")
    else:
        print("⚠️ Thời gian quá nhỏ, không tính được TPS")

# ---- chạy thử ----
if __name__ == "__main__":
    benchmark_api(DATASET_FOLDER, concurrency=CONCURRENCY)
