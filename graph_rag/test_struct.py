"""
Primary entry point for the RAG app. Integrates other RAG functionality into a UI
"""
import os
import sys

sys.path.append('/Users/zdx/llm/graphrag_lang_neo4j')

from langchain_community.llms.moonshot import Moonshot
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from graph_rag.graph import GraphBuilder
from graph_rag.rag import HuggingFaceEmbeddings
from graph_rag.rag import BGEM3Reranker


from graph_rag.entities import Entities
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting objects, person, organization, " +
            "or business entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)


# Neo4j Client Setup
# os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["MOONSHOT_API_KEY"] = "sk-npGQpPfAFwepce8sQnxEh2c2KJxZPb6LPSS6mvH361uMKZPu"

# online 
os.environ["NEO4J_URI"] = "neo4j+s://ce0ac75d.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LyuGyjYGDBM66DaKahd4dZNLIubyanxkaO3FLmaCF70"
os.environ["MOONSHOT_API_KEY"] = "sk-npGQpPfAFwepce8sQnxEh2c2KJxZPb6LPSS6mvH361uMKZPu"
os.environ["DASHSCOPE_API_KEY"] = "sk-9701adf350c2443680ba2486a765ba53"

os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'prelude-culture-corner-panel-lava-5546'
os.environ["NEO4J_URI"] = 'bolt://localhost:7687'
os.environ["GROQ_API_KEY"] = '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'


from langchain_community.chat_models.moonshot import MoonshotChat
# llm  = Moonshot()
llm = MoonshotChat()
# llm_groq = ChatGroq(temperature=0, model_name="llama3-groq-8b-8192-tool-use-preview") # model_name="mixtral-8x7b-32768")
# llm = llm_groq
#from vllm import LLM
#prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
#llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
#outputs = llm.generate(prompts)  # Generate texts from the prompts.
# test with glm GPT4ALL 
# from gpt4all import GPT4All
# glm_ = '/Users/zdx/llm/glm-4-9b-chat.Q2_K.gguf'
# llm_model = GPT4All(glm_)
# with llm_model.chat_session():
#     print(llm_model.generate("11.11213 和11.91111 哪个大", max_tokens=1024))
# chatglm.cpp run as fellow ERROR!
# llm  = ChatOpenAI(api_key=os.environ["MOONSHOT_API_KEY"],base_url="http://127.0.0.1:8000/v1")
# Tool choice {'type': 'function', 'function': {'name': 'any'}} was specified, but the only provided tool was Entities.
# chain_test =  prompt | llm.with_structured_output(Entities)  # no surpport!!!   Tool choice {'type': 'function', 'function': {'name': 'any'}} was specified, but the only provided tool was Entities.



device = 'mps'
# reranker
# reranker = BGEM3Reranker('/Users/zdx/llm/bge-reranker-v2-m3', device=device)
# bge_m3emb = CustomBgeEmbedding('/Users/zdx/llm/bge-m3/') #'BAAI/bge-m3',  not work correct for encoder_documents

# huggingface_embeddings = HuggingFaceEmbeddings('/Users/zdx/llm/bge_large_zh/')
# huggingface_embeddings = HuggingFaceEmbeddings('/Users/zdx/llm/bge-m3/', device)
# entities = chain_test.invoke({"question": '张飞是哪里人，身形如何'})


from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

class Person(BaseModel):
    """Information about a person."""
    name: str = Field(..., description="人的名字")
    height_in_meters: float = Field(
        ..., description="人的身高单位 米."
    )
class People(BaseModel):
    """Identifying information about all people in a text."""
    people: List[Person]


def  extract_chain(pydantic_object):
    # Set up a parser
    parser = PydanticOutputParser(pydantic_object=pydantic_object)
    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                "回答用户的问题. 将输出包装为 `json` 格式\n{format_instructions}",
            ),
            ("human", "{query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    # print(prompt.invoke(query).to_string())
    chain = prompt | llm | parser
    return chain

test_chain = extract_chain(Entities)
query = "安娜Anna  23 岁有6米高"
res = test_chain.invoke({"query": query})
print(f'query: {query} extract : {res}')

test_chain = extract_chain(Person)
res = test_chain.invoke({"query": query})
print(f'query: {query} extract : {res}')

