import requests
import json

# 2. 요청 보낼 url 주소
url = "http://localhost:8000/predict/"

# 3. 같이 보낼 데이터 작성
data = {"text": "비가 오는 날 출출할 때 가기 좋은 식당이에요"}
        
# 4. post로 API서버에 요청보내기
res = requests.post(url, data=json.dumps(data))

# # 5. 결과 확인하기
print(res.text)