import pandas as pd
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import redis
from redis.commands.search.query import Query

import time
import numpy as np

df6 = pd.read_csv("df_final_v6.csv")
df6["int_index"] = df6["index"].astype("int")

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
    embeddings = model.encode([targetText]).astype(np.float32).tolist()

    client = redis.Redis(host="54.172.35.91", port=6379, decode_responses=True)

    if(range_start == 20):
        converted_range_start = "0"
        if(range_end == 20):
            converted_range_end = "300"
        if(range_end == 40):
            converted_range_end = "600"
        if(range_end == 60):
            converted_range_end = "1200"
        if(range_end == 80):
            converted_range_end = "+INF"

    if(range_start == 40):
        converted_range_start = "600"
        if(range_end == 40):
            converted_range_start = "450"
            converted_range_end = "750"
        if(range_end == 60):
            converted_range_end = "1200"
        if(range_end == 80):
            converted_range_end = "+INF"

    if(range_start == 60):
        converted_range_start = "600"
        if(range_end == 60):
            converted_range_start = "900"
            converted_range_end = "1500"
        if(range_end == 80):
            converted_range_end = "+INF"
            
    if(range_start == 80):
        converted_range_start = "1200"
        if(range_end == 80):
            converted_range_end = "+INF"

    print("start0")
    start = time.time()
    pre_q0 = "((-@category0:\"식당 아님\") (@reqtime:["+converted_range_start+' '+converted_range_end+"]))=>[KNN 1033 @vector $query_vector AS vector_score]"
    query0 = (
        Query(pre_q0)
        .sort_by('vector_score')
        .return_fields('idx', 'vector_score')
        .dialect(2)
        .paging(0, 1034)
    )
    res0 = client.ft('idx:restaurant_vss').search(
        query0,
        {
        'query_vector': np.array(embeddings, dtype=np.float32).tobytes()
        }
    ).docs
    # print(res0)
    str_res0 = str(res0).replace("Result{15 total, docs: ", '').replace("Document", '')
    dict_res0 = eval(str_res0)
    end = time.time()
    print(f"{end - start:.5f} sec")

    print("start1")
    start = time.time()
    pre_q1 = "((-@category0:\"식당 아님\") (@reqtime:["+converted_range_start+' '+converted_range_end+"]))=>[KNN 1033 @category_vector $query_vector AS category_vector_score]"
    query1 = (
        Query(pre_q1)
        .sort_by('category_vector_score')
        .return_fields('idx', "category_vector_score")
        .dialect(2)
        .paging(0, 1034)
    )
    res1 = client.ft('idx:restaurant_vss').search(
        query1,
        {
        'query_vector': np.array(embeddings, dtype=np.float32).tobytes()
        }
    ).docs
    # print(res1)
    str_res1 = str(res1).replace("Result{15 total, docs: ", '').replace("Document", '')
    dict_res1 = eval(str_res1)
    end = time.time()
    print(f"{end - start:.5f} sec")

    print("start_return")
    start = time.time()
    # from collections import Counter
    # result = dict(Counter(dict_res0)+Counter(dict_res1))

    df0 = pd.DataFrame.from_dict(data=dict_res0).reset_index()
    print(df0.head(), len(df0))

    df1 = pd.DataFrame.from_dict(data=dict_res1).reset_index()
    print(df1.head(), len(df1))

    df2 = pd.merge(left=df0, right=df1, how="inner", on="idx")[["idx", "vector_score", "category_vector_score"]]
    df2["mixed_score"] = df2["vector_score"].astype("float")*df2["category_vector_score"].astype("float")
    df2 = df2.sort_values(by="mixed_score", ascending=True)
    df2["int_idx"] = df2["idx"].astype("int")
    print(df2.head(), len(df2))

    df_ret = pd.merge(df2[["int_idx", "mixed_score"]], df6[["int_index","name","position","total_distance","total_time","category3"]], how="inner", left_on="int_idx", right_on="int_index")

    categoryDict = {
        "해물 요리": "seafood",
        "한식당": "korean",
        "일식당": "japanese",
        "양식": 'western',
        "고기 요리": "beef",
        "카페": "coffee",
        "식당": "restaurant",
        "디저트": "boonsik", # 수정 필요
        "햄버거": "hamburger",
        "분식": "boonsik",
        "치킨": "chicken",
        "호프": "beer",
        "피자": 'pizza',
        "중국집": "chinese",
        "베이커리": 'bakery',
        "아시안 음식": 'asian',
        "야채 요리": 'salad',
        "주류": 'soju',
    }

    # ['해물 요리', '한식당', '일식당', nan, '양식', '고기 요리', '카페', '식당', '디저트',
    #    '햄버거', '식당 아님', '분식', '치킨', '호프', '피자', '중국집', '베이커리', '아시안 음식',
    #    '야채 요리', '주류']

    ret_val = []
    for idx in range(len(df_ret[:5])):
        name = df_ret.iloc[idx]["name"]
        vector_score = df_ret.iloc[idx]["mixed_score"]
        addr = df_ret.iloc[idx]["position"]
        dist = df_ret.iloc[idx]["total_distance"]
        reqtime = df_ret.iloc[idx]["total_time"]
        category = df_ret.iloc[idx]["category3"]

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