import requests
import json

url = "http://localhost:8000/predict/"

data = {"text": "비가 오는 날 출출할 때 가기 좋은 식당이에요"}
        
res = requests.post(url, data=json.dumps(data))

print(res.text)