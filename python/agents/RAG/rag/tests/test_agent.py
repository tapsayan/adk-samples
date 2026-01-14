import os
import pytest
from rag.agent import root_agent

def test_agent_initialization(monkeypatch):
    """Tests that the root_agent can be initialized with a mocked RAG_CORPUS."""
    monkeypatch.setenv("RAG_CORPUS", "projects/mock-project/locations/us-central1/ragCorpora/mock-corpus")
    # Attempt to access an attribute or method to force initialization, but avoid actual API calls.
    # The goal is to ensure the agent creation itself does not fail due to missing env vars.
    assert root_agent.name == 'ask_rag_agent'
