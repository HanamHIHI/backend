from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import chromadb

import json
import time
import pandas as pd
import numpy as np

df7 = pd.read_csv("df_final_v7.csv")
df7["int_index"] = df7["index"].astype("int")
with open("walking_routes.json", 'r', encoding="utf-8") as f:
    working_routes = json.load(f)
with open("car_routes.json", 'r', encoding="utf-8") as f:
    car_routes = json.load(f)

client_review = chromadb.HttpClient(host='44.203.86.11', port=8000)
client_category = chromadb.HttpClient(host='44.203.86.11', port=8000)
collection_review = client_review.get_collection(name="review_db")
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
    print("start")
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
    print(converted_range_start, converted_range_end)

    # 데이터 쿼리
    review_results = collection_review.query(
        query_embeddings=embeddings,
        where={"$and": [{"category0": {"$ne": "식당 아님"}}, {"category0": {"$ne": ''}}]},
        n_results=9930,
    )

    results_category = collection_category.query(
        query_embeddings=embeddings,
        n_results=19
    )

    review_metadatas = review_results["metadatas"][0]
    review_scores = review_results["distances"][0]
    category_metadatas = results_category["metadatas"][0]
    category_scores = results_category["distances"][0]

    categoryDict = {
        "해물 요리": "seafood",
        "한식당": "korean",
        "일식당": "japanese",
        "양식": 'western',
        "고기 요리": "meat",
        "카페": "coffee",
        "식당": "restaurant",
        "디저트": "dessert",
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

    review_data = []
    for idx in range(len(review_metadatas)):
        name = review_metadatas[idx]["name"]
        vector_score = review_scores[idx]
        category = review_metadatas[idx]["category0"]

        review_data.append({
            "name": str(name),
            "vector_score": vector_score,
            "category0": str(category),
        })

    category_data = []
    for idx in range(len(category_metadatas)):
        category = category_metadatas[idx]["category0"]
        category_score = category_scores[idx]

        category_data.append({
            "category_score": category_score,
            "category0": str(category),
        })

    df_scores_review = pd.DataFrame(review_data, columns=["name", "vector_score", "category0"])
    df_scores = pd.pivot_table(df_scores_review, index=["name", "category0"], values="vector_score", aggfunc="mean").reset_index()
    df_scores_category = pd.DataFrame(category_data, columns=["category_score", "category0"])

    df_total_score = pd.merge(df_scores, df_scores_category, how="inner", on="category0")
    df_total_score["total_score"] = df_total_score["vector_score"]+df_total_score["category_score"]
    df = pd.merge(df7, df_total_score, how="inner", left_on="name", right_on="name").sort_values(by="total_score", ascending=True)[["name", "total_score", "position", "total_distance", "total_time", "category0"]]
    df_ranged = df.loc[(df["total_time"] >= int(converted_range_start)) & (df["total_time"] <= int(converted_range_end))]

    ret_val = []
    for idx in range(5):
        name = df_ranged.iloc[idx]["name"]
        vector_score = df_ranged.iloc[idx]["total_score"]
        addr = df_ranged.iloc[idx]["position"]
        dist = df_ranged.iloc[idx]["total_distance"]
        reqtime = df_ranged.iloc[idx]["total_time"]
        category = df_ranged.iloc[idx]["category0"]

        route = ''
        if(int(reqtime) <= 900):
            for route in working_routes["routes"]:
                if(route["name"] == name):
                    route = route["route"]
                    break
        else:
            for route in car_routes["routes"]:
                if(route["name"] == name):
                    route = route["route"]
                    break
        
        ret_val.append({
            "idx": idx+1,
            "name": str(name),
            "vector_score": vector_score,
            "addr": str(addr),
            "dist": int(dist),
            "reqtime": int(reqtime),
            "category0": categoryDict[str(category)],
            "route": route,
        })
    dict_res ={"vals": ret_val}   
    
    end = time.time()
    print(f"{end - start:.5f} sec") 

    return dict_res