import ast
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tiktoken


# =========================
# Data Models
# =========================

class Relationship:
    def __init__(self, relat_type, target):
        self.relat_type = relat_type
        self.target = target


class Chunk:
    def __init__(self, name, start_line, end_line, code, relationships=None):
        self.name = name
        self.start_line = start_line
        self.end_line = end_line
        self.code = code
        self.relationships = relationships if relationships is not None else []


# =========================
# Chunking باستخدام AST
# =========================

def chunk_file(file_path, max_tokens=1500):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        lines = content.splitlines()

        chunks = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):

                code = "\n".join(lines[node.lineno - 1: node.end_lineno])

                chunk = Chunk(
                    name=f"{file_path}:{node.name}",
                    start_line=node.lineno,
                    end_line=node.end_lineno,
                    code=code,
                    relationships=[]
                )

                # =========================
                # Relationships: has_method + inherits
                # =========================
                if isinstance(node, ast.ClassDef):
                    token_count = len(enc.encode(code))
                    if token_count > max_tokens:
                        for child in ast.iter_child_nodes(node):
                            if isinstance(child, ast.FunctionDef):
                                chunk = Chunk(
                                    name=f"{file_path}:{node.name}:{child.name}",
                                    start_line=child.lineno,
                                    end_line=child.end_lineno,
                                    code="\n".join(lines[child.lineno - 1: child.end_lineno]),
                                    relationships=[]
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
                            if isinstance(child, ast.FunctionDef):
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
                if isinstance(node, ast.FunctionDef):
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

        return chunks


# =========================
# Build Graph
# =========================

def build_graph(chunks):
    G = nx.DiGraph()

    for chunk in chunks:
        G.add_node(
            chunk.name,
            code=chunk.code,
            start_line=chunk.start_line,
            end_line=chunk.end_line
        )

        for r in chunk.relationships:
            G.add_edge(chunk.name, r.target, relat_type=r.relat_type)

    return G


# =========================
# Embeddings
# =========================
enc = tiktoken.get_encoding("cl100k_base")
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    embeddings = {}

    for chunk in chunks:
        vec = model.encode(chunk.code)
        embeddings[chunk.name] = vec

    return embeddings


# =========================
# Similarity Search
# =========================

def similarity_search(query, embeddings, k=5):
    query_vec = model.encode(query)

    scores = []

    for name in embeddings:
        chunk_vec = embeddings[name]

        sim = cosine_similarity(
            query_vec.reshape(1, -1),
            chunk_vec.reshape(1, -1)
        )[0][0]

        scores.append((sim, name))

    scores = sorted(scores, reverse=True)

    return [name for _, name in scores[:k]]


# =========================
# Graph Traversal (GraphRAG)
# =========================

def get_context(top_k_chunks, G):
    context = set(top_k_chunks)

    for chunk in top_k_chunks:
        neighbors = G.neighbors(chunk)
        context.update(neighbors)

    return list(context)


# =========================
# Full Pipeline
# =========================

def graphrag_pipeline(file_paths, query):

    all_chunks = []

    # 1. Chunk all files
    for path in file_paths:
        chunks = chunk_file(path,max_tokens=50)
        all_chunks.extend(chunks)

    # 2. Build graph
    G = build_graph(all_chunks)

    # 3. Embed
    embeddings = embed_chunks(all_chunks)

    # 4. Similarity search
    top_chunks = similarity_search(query, embeddings, k=1)

    # 5. Graph expansion
    context_nodes = get_context(top_chunks, G)

    # 6. Get actual code
    context_code = []
    for node in context_nodes:
        if node in G.nodes and "code" in G.nodes[node]:
             context_code.append(G.nodes[node]["code"])


    return "\n\n".join(context_code)


# =========================
# Example Usage
# =========================

"""if __name__ == "__main__":
    files = ["bigclass.py"]  # ضع ملفاتك هنا
    query = "How does the Dog class work?"

    context = graphrag_pipeline(files, query)

    print("====== Retrieved Context ======")
    print(context)"""

if __name__ == "__main__":
    chunks = chunk_file("bigclass.py", max_tokens=50)
    for chunk in chunks:
        print(f"Chunk: {chunk.name} | lines {chunk.start_line}-{chunk.end_line}")
        print(f"Relationships: {[(r.relat_type, r.target) for r in chunk.relationships]}")
        print("---")