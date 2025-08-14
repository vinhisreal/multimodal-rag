from multimodal_rag import AdvancedMultimodalRAGSystem, AdvancedRAGConfig

def main():
    """Advanced RAG demonstration"""
    print("🚀 Advanced Multimodal RAG Demonstration")
    print("=" * 50)
    
    # Advanced configuration
    config = AdvancedRAGConfig()
    config.persist_directory = "./advanced_rag_db"
    config.use_reranking = True
    config.rerank_method = 'hybrid'
    config.use_record_manager = True
    config.enable_monitoring = True
    
    # Initialize advanced system
    advanced_rag = AdvancedMultimodalRAGSystem(config)
    
    # Show system capabilities
    print("\n📊 System Features:")
    print(f"   🔍 Hybrid Search: {'✅' if config.use_hybrid_search else '❌'}")
    print(f"   🎯 Reranking: {'✅' if config.use_reranking else '❌'}")
    print(f"   📁 Record Manager: {'✅' if config.use_record_manager else '❌'}")
    print(f"   📊 Monitoring: {'✅' if config.enable_monitoring else '❌'}")
    
    # Demo queries with different features
    demo_queries = [
        ("Basic query", "What is artificial intelligence?"),
        ("Complex query", "Compare the performance metrics and explain the trends"),
        ("Multimodal query", "Show me the charts and explain what they mean"),
        ("Vietnamese query", "Tóm tắt các kết quả chính trong báo cáo")
    ]
    
    print(f"\n🔍 Testing {len(demo_queries)} queries:")
    print("-" * 50)
    
    for query_type, query in demo_queries:
        print(f"\n🔸 {query_type}: {query}")
        
        # Query with reranking
        response = advanced_rag.query(query, n_results=3, use_reranking=True)
        
        print(f"   ⏱️  Time: {response.processing_time:.2f}s")
        print(f"   🎯 Rerank: {response.metadata.get('rerank_method', 'none')}")
        print(f"   📚 Sources: {len(response.source_elements)}")
        print(f"   🤖 Answer preview: {response.answer[:100]}...")
    
    # Show system statistics
    if hasattr(advanced_rag, 'get_system_stats'):
        print(f"\n📈 System Statistics:")
        stats = advanced_rag.get_system_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
    
    print(f"\n🎉 Advanced RAG demonstration completed!")

if __name__ == "__main__":
    main()