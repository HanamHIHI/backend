import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(host='44.203.86.11', port=8000)
print("client setting ok")

# 원격 컬렉션 접근
collection = client.get_collection(name="restaurant_db")

print(collection.count())

# 데이터 쿼리
query_embedding = [0.15] * 768
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=12
)

print(results)
