from config import embedding_model , OPENROUTER_API_KEY
from openai import OpenAI


def similarity_search(query, driver, k=5):
    query_vec = embedding_model.encode(query)
    with driver.session() as session:
        results = session.run(
            """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
            YIELD node, score
            RETURN node.name, score
            """,
            k=k,
            embedding=query_vec.tolist(),
        )
        return [r["node.name"] for r in results]


def get_context(top_k_chunks, driver):
    context = set(top_k_chunks)
    with driver.session() as session:
        for chunk_name in top_k_chunks:
            results = session.run(
                "MATCH (a:Chunk {name: $name})-[:RELATES]->(b) RETURN b.name",
                name=chunk_name,
            )
            for res in results:
                context.add(res["b.name"])
    return list(context)


def _generate_followup(query, chunks,driver):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model="xiaomi/mimo-v2-pro",
        messages=[
            {"role": "system", "content": f"You are a code analysis assistant. Given a query and some           code context, generate ONE follow-up question that would help explore the codebase deeper.\nContext: {chunks}"},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content




def drift_search(query, driver, k=5, num_hops=3):
    all_context = set()
    current_query = query
    
    for hop in range(num_hops):
        top_chunks = similarity_search(current_query, driver, k)
        all_context.update(top_chunks)
        
        neighbors = get_context(top_chunks, driver)
        all_context.update(neighbors)
        
        current_query = _generate_followup(current_query, top_chunks, driver)
    
    return list(all_context)