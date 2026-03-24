import ast
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tiktoken
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j import TrustAll
from openai import OpenAI
import os
load_dotenv()

# =========================
# Data Models
# =========================




class Relationship:
    def __init__(self, relat_type, target):
        self.relat_type = relat_type
        self.target = target


class Chunk:
    def __init__(self, name, start_line, end_line, code, relationships=None,docstring=None):
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.code = code
        self.relationships = relationships if relationships is not None else []
        self.docstring = docstring


# =========================
# Chunking باستخدام AST
# =========================

def chunk_file(file_path, max_tokens=1500):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        lines = content.splitlines()

        chunks = []
        import_lines = []
        global_lines = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                code = "\n".join(lines[node.lineno - 1: node.end_lineno])
                import_lines.append(code)

            elif isinstance(node,ast.Assign):
                code = "\n".join(lines[node.lineno - 1: node.end_lineno])
                global_lines.append(code)

            elif isinstance(node, (ast.FunctionDef, ast.ClassDef,ast.AsyncFunctionDef)):

                code = "\n".join(lines[node.lineno - 1: node.end_lineno])

                chunk = Chunk(
                    name=f"{file_path}:{node.name}",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    code=code,
                    relationships=[],
                    docstring=ast.get_docstring(node)
                )

                # =========================
                # Relationships: has_method + inherits
                # =========================
                if isinstance(node, ast.ClassDef):
                    token_count = len(enc.encode(code))
                    if token_count > max_tokens:
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child,( ast.FunctionDef,ast.AsyncFunctionDef)):
                                chunk = Chunk(
                                    name=f"{file_path}:{node.name}:{child.name}",
                                    start_line=child.lineno,
                                    end_line=child.end_lineno,
                                    code="\n".join(lines[child.lineno - 1: child.end_lineno]),
                                    relationships=[],
                                    docstring= ast.get_docstring(child)
                                )
                                chunks.append(chunk)
                                for n in ast.walk(child):
                                    if isinstance(n, ast.Call):
                                        try:
                                            target = n.func.id
                                        except AttributeError:
                                            try:
                                                target = n.func.attr
                                            except AttributeError:
                                                continue
                                        rel = Relationship("calls", target)
                                        chunk.relationships.append(rel)
                        continue

                    else:

                        # methods
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child,( ast.FunctionDef,ast.AsyncFunctionDef)):
                                rel = Relationship("has_method", target=f"{file_path}:{child.name}")
                                chunk.relationships.append(rel)

                        # inheritance
                        for base in node.bases:
                            try:
                                rel = Relationship("inherits", base.id)
                                chunk.relationships.append(rel)
                            except AttributeError:
                                pass
                            

                # =========================
                # Relationships: calls
                # =========================
                if isinstance(node,( ast.FunctionDef,ast.AsyncFunctionDef)):
                    for n in ast.walk(node):
                        if isinstance(n, ast.Call):
                            try:
                                target = n.func.id  # simple call like bark()
                            except AttributeError:
                                try:
                                    target = n.func.attr  # method call like self.bark()
                                except AttributeError:
                                    continue
                            rel = Relationship("calls", target)
                            chunk.relationships.append(rel)

                chunks.append(chunk)

        if import_lines:
            chunk = Chunk(
                name=f"{file_path}:imports",
                start_line=0,
                end_line=0,
                code="\n".join(import_lines),
                relationships=[],
                docstring=None
            )
            chunks.append(chunk)

        if global_lines:
            chunk = Chunk(
                name=f"{file_path}:globals",
                start_line=0,
                end_line=0,
                code="\n".join(global_lines),
                relationships=[],
                docstring=None
            )
            chunks.append(chunk)

        return chunks

def build_graph(chunks, driver):
    with driver.session() as session:
        for chunk in chunks:
            session.run(
                "MERGE (n:Chunk {name: $name, code: $code,start_line:$start_line,end_line:$end_line})",
                name=chunk.name,
                code=chunk.code,
                start_line = chunk.start_line,
                end_line = chunk.end_line
            )
        for chunk in chunks:
            for r in chunk.relationships:
                session.run(
                    "MATCH (a:Chunk {name: $source}), (b:Chunk) WHERE b.name ENDS WITH $target CREATE (a)-[:RELATES {type: $rel_type}]->(b)",
                    source=chunk.name,
                    target=r.target,
                    rel_type=r.relat_type
            )
                
model = SentenceTransformer("all-MiniLM-L6-v2")
enc = tiktoken.get_encoding("cl100k_base")

def embd_and_store(chunks,driver):

    with driver.session() as session:
        session.run("""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (n:Chunk) ON n.embedding
            OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}
            """)
        for chunk in chunks:
            vec = model.encode(chunk.code)
            
            session.run(
            "MATCH (n:Chunk {name: $name}) SET n.embedding = $embedding",
            name=chunk.name,
            embedding=vec.tolist()
            )


def similarity_search(query, driver, k=5):
    query_vec = model.encode(query)
    with driver.session() as session:
        results = session.run(
            """CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
            YIELD node, score
            RETURN node.name, score""",
            k=k,
            embedding=query_vec.tolist()
        )
        return [r["node.name"] for r in results]
    

def get_context(top_k_chunks, driver):
    context = set(top_k_chunks)
    with driver.session() as session:
        for chunk_name in top_k_chunks:
            results = session.run(
                "MATCH (a:Chunk {name: $name})-[:RELATES]->(b) RETURN b.name",
                name=chunk_name
            )
            for res in results:
                context.add(res["b.name"])
    return list(context)



def graphrag_pipeline(file_paths, query,driver):
    all_chunks = []

    # 1. Chunk all files
    for path in file_paths:
        chunks = chunk_file(path,max_tokens=50)
        all_chunks.extend(chunks)

    # 2. Build graph
    build_graph(all_chunks,driver)

    # 3. Embed
    embd_and_store(all_chunks,driver)

    # 4. Similarity search
    top_chunks = similarity_search(query, driver)

    # 5. Graph expansion
    context_nodes = get_context(top_chunks, driver)

    context_code = []
    with driver.session() as session:
        for node in context_nodes:
            result = session.run(
                "MATCH (n:Chunk {name: $name}) RETURN n.code",
                name=node
            )
            for r in result:
                context_code.append(r["n.code"])

    def generate_response(query, context):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        response = client.chat.completions.create(
            model="xiaomi/mimo-v2-pro",
            messages=[
                {"role": "system", "content": f"You are a coding assistant. Use this context:\n{context}"},
                {"role": "user", "content": query}
            ]
        )
        
        return response.choices[0].message.content

    context = "\n\n".join(context_code)
    return generate_response(query, context)


url = os.getenv("NEO4J_URI")
password = os.getenv("NEO4J_PASSWORD")
username = os.getenv("NEO4J_USERNAME")
if __name__ == "__main__":
    
    driver = GraphDatabase.driver(
        url,
        auth=(username, password),
    )
    
    files = ["bigclass.py"]
    query = "How does the Dog class bark?"
    
    context = graphrag_pipeline(files, query, driver)
    print("====== Retrieved Context ======")
    print(context)
    
    driver.close()