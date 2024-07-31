import requests

image_path = './testset/images/0001176_as-0066940_jpg.rf.cb450b3d48f2913b0df9568d9238ec1e.jpg'
url = 'http://127.0.0.1:5000/segment'

with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.status_code)
