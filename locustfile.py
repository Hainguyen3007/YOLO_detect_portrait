from locust import HttpUser, task, between

class DetectUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def detect(self):
        # gửi 1 request POST với file ảnh
        with open(r"image/cccd_MTD.jpg", "rb") as f:
            files = {"file": f}
            response = self.client.post("/detect", files=files)

            # in log khi request lỗi
            if response.status_code != 200:
                print(f"❌ Lỗi {response.status_code}: {response.text}")
