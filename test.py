from milvus_utilis import search_similar_chunks

# Your test query
query = "Điều khoản chấm dứt hợp đồng là gì?"

# Call the search function
results = search_similar_chunks(query, top_k=3)

# Print the top-k similar chunks
for i, result in enumerate(results, 1):
    print(f"🔍 Kết quả {i}:")
    print(f"📌 Điểm tương đồng: {result['score']:.4f}")
    print(f"📄 Đoạn văn: {result['chunk']}\n")
