import pytest

from embeddings_rag import AdvancedRAG
from llm_interface import LLMFactory


@pytest.fixture(scope="session")
def llm():
    config = {
        "llm_provider": "generic",
        "llm_model": "gpt-4o-mini",
        "api_base": "http://localhost:1234/v1",
    }
    return LLMFactory.create(config)


@pytest.fixture(scope="session")
def rag(llm):
    rag = AdvancedRAG(llm=llm, embedding_model_name="BAAI/bge-m3")
    return rag
