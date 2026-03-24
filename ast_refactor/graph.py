from config import embedding_model


def build_graph(chunks, driver):
    with driver.session() as session:
        for chunk in chunks:
            session.run(
                "MERGE (n:Chunk {name: $name, code: $code, start_line: $start_line, end_line: $end_line})",
                name=chunk.name,
                code=chunk.code,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
        for chunk in chunks:
            for r in chunk.relationships:
                session.run(
                    """
                    MATCH (a:Chunk {name: $source}), (b:Chunk)
                    WHERE b.name ENDS WITH $target
                    CREATE (a)-[:RELATES {type: $rel_type}]->(b)
                    """,
                    source=chunk.name,
                    target=r.target,
                    rel_type=r.relat_type,
                )


def embd_and_store(chunks, driver):
    with driver.session() as session:
        session.run("""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (n:Chunk) ON n.embedding
            OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
        """)
        for chunk in chunks:
            vec = embedding_model.encode(chunk.code)
            session.run(
                "MATCH (n:Chunk {name: $name}) SET n.embedding = $embedding",
                name=chunk.name,
                embedding=vec.tolist(),
            )