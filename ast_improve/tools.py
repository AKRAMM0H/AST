import os
from chunker import chunk_file
from graph import build_graph, embd_and_store
from search import similarity_search , get_context,drift_search
from pipeline import generate_response
import hashlib

SKIP_DIRS = {'.venv', 'venv', '__pycache__', '.git', 'dist', 'build', '.mypy_cache', '.pytest_cache', '.tox'}

def _get_stored_hash(file_path, driver):
    with driver.session() as session:
        result = session.run(
            "MATCH (f:File {path: $path}) RETURN f.last_parsed_hash",
            path=file_path
        )
        record = result.single()
        return record["f.last_parsed_hash"] if record else None

def index_codebase(path, driver):
    all_chunks = []
    all_file_nodes = []
    
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root,file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    current_hash = hashlib.md5(content.encode()).hexdigest()
                stored_hash = _get_stored_hash(file_path, driver)
                if stored_hash == current_hash:
                    continue
                chunks, file_node = chunk_file(file_path)
                if not chunks:
                    continue
                all_chunks.extend(chunks)
                all_file_nodes.append(file_node)
    
    build_graph(all_chunks, all_file_nodes, driver)
    embd_and_store(all_chunks,driver)



def query(question,driver):
    top_chunks = similarity_search(question, driver)

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
    return generate_response(question, context)



def clear_index(driver):
    with driver.session() as session:
        session.run(
            "MATCH (n) DETACH DELETE n"
        )


def update_file(file_path, driver):
    # 1. Delete existing nodes for this file
    with driver.session() as session:
        session.run(
            "MATCH (n) WHERE n.name STARTS WITH $file_path OR n.path = $file_path DETACH DELETE n",
            file_path=file_path,
        )
    
    # 2. Re-chunk the file
    chunks, file_node = chunk_file(file_path)
    
    # 3. Rebuild graph and embeddings
    build_graph(chunks, [file_node], driver)
    embd_and_store(chunks, driver)


def drift_query(question,driver):

    context_nodes = drift_search(question, driver)

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
    return generate_response(question, context)