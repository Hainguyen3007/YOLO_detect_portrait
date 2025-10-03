import requests

url = "http://127.0.0.1:8000/detect"
files = {"file": open("../image/cccd_MTD.jpg", "rb")}
response = requests.post(url, files=files)

with open("../output/portrait.png", "wb") as f:
    f.write(response.content)
