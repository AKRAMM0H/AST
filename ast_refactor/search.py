from config import embedding_model


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