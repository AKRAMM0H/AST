from tools import index_codebase, query,clear_index , drift_query
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
from config import NEO4J_URI, NEO4J_PASSWORD, NEO4J_USERNAME, OPENROUTER_API_KEY


load_dotenv()

if __name__ == "__main__":
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        max_connection_lifetime=3600,
        keep_alive=True
    )
    clear_index(driver)

    index_codebase("D:\AST\ast_improve", driver)
    # Local
    answer1 = query("explain this codebase ", driver)
    print("LOCAL:", answer1)

    # Drift  
    answer2 = drift_query("explin this codebase", driver)
    print("\n")
    print("DRIFT:", answer2)
    
  
    
    driver.close()