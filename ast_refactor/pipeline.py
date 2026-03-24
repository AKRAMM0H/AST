from openai import OpenAI
from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_PASSWORD, NEO4J_USERNAME, OPENROUTER_API_KEY
from chunker import chunk_file
from graph import build_graph, embd_and_store
from search import similarity_search, get_context


def generate_response(query, context):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model="xiaomi/mimo-v2-pro",
        messages=[
            {"role": "system", "content": f"You are a coding assistant. Use this context:\n{context}"},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


def graphrag_pipeline(file_paths, query, driver):
    # 1. Chunk all files
    all_chunks = []
    for path in file_paths:
        all_chunks.extend(chunk_file(path, max_tokens=50))

    # 2. Build graph
    build_graph(all_chunks, driver)

    # 3. Embed and store
    embd_and_store(all_chunks, driver)

    # 4. Similarity search
    top_chunks = similarity_search(query, driver)

    # 5. Graph expansion
    context_nodes = get_context(top_chunks, driver)

    # 6. Fetch code for context nodes
    context_code = []
    with driver.session() as session:
        for node in context_nodes:
            result = session.run(
                "MATCH (n:Chunk {name: $name}) RETURN n.code",
                name=node,
            )
            for r in result:
                context_code.append(r["n.code"])

    context = "\n\n".join(context_code)
    return generate_response(query, context)


if __name__ == "__main__":
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    files = ["bigclass.py"]
    query = "How does the Dog class bark?"

    result = graphrag_pipeline(files, query, driver)
    print("====== Retrieved Context ======")
    print(result)

    driver.close()