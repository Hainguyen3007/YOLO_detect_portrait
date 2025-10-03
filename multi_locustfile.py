import os
import random
import time
from locust import HttpUser, task, between, events

# 📂 Thư mục chứa ảnh test
IMG_DIR = "test_tps"

# Lấy danh sách ảnh hợp lệ
images = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

BATCH_SIZE = 5  # số ảnh gửi mỗi "batch"

# Biến toàn cục để thống kê TPS theo ảnh
total_images = 0
start_time = None


class DetectUser(HttpUser):
    wait_time = between(1, 2)  # nghỉ giữa các batch

    @task
    def detect_batch(self):
        global total_images, start_time

        if start_time is None:
            start_time = time.time()

        # chọn ngẫu nhiên BATCH_SIZE ảnh
        batch_imgs = random.sample(images, min(BATCH_SIZE, len(images)))

        for img_path in batch_imgs:
            with open(img_path, "rb") as f:
                files = {"file": f}
                response = self.client.post("/detect", files=files)

                if response.status_code != 200:
                    print(f"❌ Lỗi {response.status_code} với {img_path}: {response.text}")
                else:
                    total_images += 1
                    print(f"✅ Thành công với {os.path.basename(img_path)} trong batch")


# 📊 Hàm hook của Locust để log TPS theo ảnh/giây khi test kết thúc
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    global total_images, start_time
    if start_time is not None and total_images > 0:
        elapsed = time.time() - start_time
        tps_images = total_images / elapsed
        print("\n================ TPS REPORT (theo ảnh) ================\n")
        print(f"Tổng số ảnh xử lý: {total_images}")
        print(f"Tổng thời gian chạy: {elapsed:.2f} giây")
        print(f"TPS (ảnh/giây): {tps_images:.2f}")
        print("\n=======================================================\n")
