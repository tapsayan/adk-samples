# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import uuid
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag
from openinference.instrumentation import using_session
from rag.tracing import instrument_adk_with_arize
from unittest.mock import MagicMock
from google.genai.types import GenerateContentResponse


_ = instrument_adk_with_arize()

from dotenv import load_dotenv
from .prompts import return_instructions_root

load_dotenv()

# Use a mock for testing to avoid actual API calls
if os.environ.get("PYTEST_RUNNING"):
    ask_vertex_retrieval = MagicMock(spec=VertexAiRagRetrieval)
    ask_vertex_retrieval.name = "retrieve_rag_documentation"
    ask_vertex_retrieval.description = "A mock retrieval tool."
    tools_to_use = [ask_vertex_retrieval]

    mock_model = MagicMock()
    mock_response_dict = {'candidates': [{'content': {'parts': [{'text': 'This is a mocked response.'}], 'role': 'model'}}]}
    mock_response = GenerateContentResponse.from_dict(mock_response_dict)

    async def mock_generate_content_async(*args, **kwargs):
        yield mock_response

    mock_model.generate_content_async = mock_generate_content_async
    model_to_use = mock_model
else:
    ask_vertex_retrieval = VertexAiRagRetrieval(
        name='retrieve_rag_documentation',
        description=(
            'Use this tool to retrieve documentation and reference materials for the question from the RAG corpus,'
        ),
        rag_resources=[
            rag.RagResource(
                # please fill in your own rag corpus
                # here is a sample rag corpus for testing purpose
                # e.g. projects/123/locations/us-central1/ragCorpora/456
                rag_corpus=os.environ.get("RAG_CORPUS")
            )
        ],
        similarity_top_k=10,
        vector_distance_threshold=0.6,
    )
    tools_to_use = [ask_vertex_retrieval]
    model_to_use = 'gemini-2.0-flash-001'


with using_session(session_id=uuid.uuid4()):
    root_agent = Agent(
        model=model_to_use,
        name='ask_rag_agent',
        instruction=return_instructions_root(),
        tools=tools_to_use,
    )
