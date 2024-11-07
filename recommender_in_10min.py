import csv
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from tqdm import tqdm 
import time

tags = ['hanam']
tag = tags[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 800 # hyper_params.
test_count = 200
batch_size = 16
epochs = 128

df3 = pd.read_csv("df_in_10min_v2.csv", encoding="utf-8")
mapper = list(df3['name'])
category_list = df3["category3"].unique()

data_df = pd.read_csv("preprocessed_urls_hanam_restaurant_real_url_review.csv", encoding="utf-8")
data_in_10min_df = data_df.loc[data_df["name"].isin(mapper)]
test_data = [[] for _ in list(range(len(mapper)))]
temp_data = []

for idx in range(len(data_in_10min_df)):
    test_data[mapper.index(str(data_in_10min_df.iloc[idx]["name"]))].append(str(data_in_10min_df.iloc[idx]["review"]))
    temp_data.append([str(data_in_10min_df.iloc[idx]["review"]), mapper.index(str(data_in_10min_df.iloc[idx]["name"]))])

test_examples = []
f = open(tag + '_test_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    test_examples.append(row)
f.close()

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

try:
    model_state_dict = torch.load("basic_model_1000" +  ".pt", map_location=device)
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
        model_state_dict = torch.load("basic_model_1000" +  ".pt", map_location=device)
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

import numpy as np

vectors = [[] for _ in list(range(len(mapper)))]
try:
    f = open(tag + '_vectors_' + str(train_count+test_count) + '_10min.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row), 0.3396976888179779, "int(temp_data[i][1])", int(temp_data[i][1]))
        # print(i, int(temp_data[i][1]))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

    print("reading vectors complete.")

except FileNotFoundError:
    f = open(tag + '_vectors_' + str(train_count+test_count) + '_10min.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in tqdm(list(range(len(mapper)))): # 각 강의평마다 주어진 강의평 벡터의 평균을 각 강의와 매핑합니다.
        for j in list(range(len(test_data[i]))):
            vector = model.encode(test_data[i][j])
            listedVector = vector.tolist()
            writer.writerow(listedVector)
        # print(i, "encoding done")

    f.close()

    f = open(tag + '_vectors_' + str(train_count+test_count) + '_10min.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

mean_vectors = []
try:
    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '_10min.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(np.double(_row))
        mean_vectors.append((np.array(floatCastedRow), i))
    f.close()

    print("reading mean vectors complete.")
except FileNotFoundError:
    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '_10min.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in tqdm(list(range(len(mapper)))):
        vector = model.encode(test_data[i])
        vector = np.mean(vector, axis=0)

        listedVector = vector.tolist()
        writer.writerow(listedVector)
    f.close()

    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '_10min.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(np.double(_row))
            # if(i == 0):
            #     print(_row, float(_row))
        mean_vectors.append((np.array(floatCastedRow), i))
    f.close()

acc = 0
hitsAt3 = 0
hitsAt5 = 0
hitsAt10 = 0
rankingBasedMetric = 0

def predict(targetText: str):
    print("Start recommending")
    start_time = time.time()

    targetVector = model.encode([targetText]) # targetVector는 데스트 할 text string의 sentence vector

    category_sim_list = []
    for category in category_list:
        v1 = model.encode([category])
        sim = util.cos_sim(targetVector, v1) # compute similarity between sentence vectors
        category_sim_list.append([category, sim])
    category_sim_df = pd.DataFrame(category_sim_list, columns=["category", "sim"])

    for i in list(range(test_count)):
        results = []
        answerList = []
        for j in list(range(len(mapper))):
            similarities = util.cos_sim(np.array(mean_vectors[j][0], dtype=np.float32), targetVector) # compute similarity between sentence vectors
            target_category = df3.loc[df3["name"]==mapper[j]]["category3"].values[0]
            # print(target_category)
            target_category_sim = category_sim_df.loc[category_sim_df["category"]==target_category]["sim"].values[0]

            results.append((j, mapper[j], float(similarities), float(target_category_sim), float(similarities)*float(target_category_sim)))
        results.sort(key = lambda x : -x[4])

    print(targetText, "에 적합한 식당은")
    print("번호", "상호명", "review_score", "category_score", "total_score")
    print("="*45)

    ret_val = []
    for idx, result in enumerate(results[:5]):
        printedString = result
        print(printedString)

        ret_val.append({
            "idx": result[0],
            "name": result[1],
            "review_score": result[2],
            "category_score": result[3],
            "total_score": result[4],
        })
    print("입니다.")

    print(f"Recommending done after {time.time() - start_time:.2f} sec")

    return {"vals": ret_val}

# predict("테스트 텍스트")