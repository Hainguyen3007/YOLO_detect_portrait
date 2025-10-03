import os
import time
import requests

API_URL = "http://127.0.0.1:8000/detect"  # endpoint API
FOLDER = "test_tps"      # 👉 thay bằng folder ảnh của bạn

def test_api_tps(folder, api_url):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("❌ Không tìm thấy ảnh trong thư mục.")
        return

    valid_count = 0
    start_time = time.time()

    for filename in image_files:
        filepath = os.path.join(folder, filename)
        with open(filepath, "rb") as f:
            files = {"file": (filename, f, "image/jpeg")}
            try:
                response = requests.post(api_url, files=files, timeout=20)
                if response.status_code == 200:
                    # Trường hợp API trả về file ảnh portrait
                    valid_count += 1
                else:
                    print(f"⚠️ Bỏ qua {filename}, status {response.status_code}")
            except Exception as e:
                print(f"⚠️ Lỗi khi gửi {filename}: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    if valid_count > 0 and total_time > 0:
        tps = valid_count / total_time
        print(f"✅ Ảnh hợp lệ: {valid_count}/{len(image_files)}")
        print(f"⏱️ Tổng thời gian: {total_time:.2f} giây")
        print(f"⚡ TPS qua API: {tps:.2f} ảnh/giây")
    else:
        print("❌ Không có ảnh hợp lệ hoặc thời gian đo bằng 0.")

if __name__ == "__main__":
    test_api_tps(FOLDER, API_URL)
