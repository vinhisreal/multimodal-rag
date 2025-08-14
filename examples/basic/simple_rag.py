from multimodal_rag import MultimodalRAGSystem, RAGConfig

def main():
    """Basic RAG example"""
    print("🚀 Simple Multimodal RAG Example")
    print("=" * 40)
    
    # Initialize system with basic configuration
    config = RAGConfig()
    config.persist_directory = "./simple_rag_db"
    config.collection_name = "simple_demo"
    
    rag_system = MultimodalRAGSystem(config)
    
    # Ingest a document (replace with your document path)
    # success = rag_system.ingest_document("path/to/your/document.pdf")
    # print(f"Document ingested: {'✅ Success' if success else '❌ Failed'}")
    
    # Query the system
    queries = [
        "What is the main topic of the document?",
        "Summarize the key findings",
        "What are the conclusions?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        response = rag_system.query(query)
        
        print(f"⏱️  Processing time: {response.processing_time:.2f}s")
        print(f"📚 Sources found: {len(response.source_elements)}")
        print(f"🤖 Answer: {response.answer}")

if __name__ == "__main__":
    main()
