import pandas as pd
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import chromadb

import time
import numpy as np

df7 = pd.read_csv("df_final_v7.csv")
df7["int_index"] = df7["index"].astype("int")

client_restaurant = chromadb.HttpClient(host='44.203.86.11', port=8000)
client_category = chromadb.HttpClient(host='44.203.86.11', port=8000)
collection_restaurant = client_restaurant.get_collection(name="restaurant_db")
collection_category = client_category.get_collection(name="category_db")
print("client setting ok")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

try:
    model_state_dict = torch.load("basic_model" +  ".pt", map_location=device)
    try:
        model.load_state_dict(model_state_dict)

    except RuntimeError:
        print("E1")
        try:
            model.load_state_dict(model_state_dict, strict=False)
        except:
            pass

except FileNotFoundError:
    print("E0")
    try:
        model_state_dict = torch.load("basic_model" +  ".pt", map_location=device)
        try:
            model.load_state_dict(model_state_dict)

        except RuntimeError:
            print("E1")
            try:
                model.load_state_dict(model_state_dict, strict=False)
            except:
                pass
    except:
        pass

print("model loading complete.")

def predict(targetText: str, range_start: int, range_end: int):
    print("start0")
    start = time.time()

    embeddings = model.encode([targetText]).astype(np.float32).tolist()
    print(len(embeddings), len(embeddings[0]))

    if(range_start == 20):
        converted_range_start = 0
        if(range_end == 20):
            converted_range_end = 300
        if(range_end == 40):
            converted_range_end = 600
        if(range_end == 60):
            converted_range_end = 1200
        if(range_end == 80):
            converted_range_end = 999999

    if(range_start == 40):
        converted_range_start = 600
        if(range_end == 40):
            converted_range_start = 450
            converted_range_end = 750
        if(range_end == 60):
            converted_range_end = 1200
        if(range_end == 80):
            converted_range_end = 999999

    if(range_start == 60):
        converted_range_start = 600
        if(range_end == 60):
            converted_range_start = 900
            converted_range_end = 1500
        if(range_end == 80):
            converted_range_end = 999999
            
    if(range_start == 80):
        converted_range_start = 1200
        if(range_end == 80):
            converted_range_end = 999999

    # 데이터 쿼리
    results_restaurant = collection_restaurant.query(
        query_embeddings=embeddings,
        # where={"$and": ["name": {"$ne": ''}, "reqtime": {"$gte": converted_range_start}, "reqtime": {"$lte": converted_range_end}]},
        where={"$and": [{"caegory0": {"$ne": "식당 아님"}}, {"$and": [{"caegory0": {"$ne": ''}}, {"$and": [{"reqtime": {"$gte": converted_range_start}}, {"reqtime": {"$lte": converted_range_end}}]}]}]},
        # where={"$and": [{"name": {"$ne": "식당 아님"}}, {"name": {"$ne": ''}}]},
        n_results=1033,
    )

    results_category = collection_category.query(
        query_embeddings=embeddings,
        n_results=19
    )

    # results = collection.peek()
    # print(results)
    ret_value = results_restaurant["metadatas"][0]
    # print(ret_value)
    scores = results_restaurant["distances"][0]
    print(scores[:5])

    print(results_category)

    categoryDict = {
        "해물 요리": "seafood",
        "한식당": "korean",
        "일식당": "japanese",
        "양식": 'western',
        "고기 요리": "meat",
        "카페": "coffee",
        "식당": "restaurant",
        "디저트": "boonsik", # 수정 필요
        "햄버거": "hamburger",
        "분식": "boonsik",
        "치킨": "chicken",
        "맥주": "beer",
        "피자": 'pizza',
        "중국집": "chinese",
        "베이커리": 'bakery',
        "아시안 음식": 'asian',
        "야채 요리": 'salad',
        "주류": 'soju',
        "nan": "restaurant",
    }

    # ['해물 요리', '한식당', '일식당', nan, '양식', '고기 요리', '카페', '식당', '디저트',
    #    '햄버거', '식당 아님', '분식', '치킨', '호프', '피자', '중국집', '베이커리', '아시안 음식',
    #    '야채 요리', '주류']

    ret_val = []
    for idx in range(len(ret_value[:5])):
        name = ret_value[idx]["name"]
        vector_score = 1
        addr = ret_value[idx]["addr"]
        dist = ret_value[idx]["dist"]
        reqtime = ret_value[idx]["reqtime"]
        category = ret_value[idx]["category0"]

        ret_val.append({
            "idx": idx+1,
            "name": str(name),
            "vector_score": vector_score,
            "addr": str(addr),
            "dist": int(dist),
            "reqtime": int(reqtime),
            "category0": categoryDict[str(category)],
        })
    dict_res ={"vals": ret_val}    
    end = time.time()
    print(f"{end - start:.5f} sec")

    return dict_res