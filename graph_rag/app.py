"""
Primary entry point for the RAG app. Integrates other RAG functionality into a UI
"""
import os
import sys
import dotenv
dotenv.load_dotenv(dotenv_path = '/Users/zdx/llm/.env')

sys.path.append('/Users/zdx/llm/graphrag_lang_neo4j')

import streamlit as st
from typing import List, Iterator, Optional, Union
from langchain_community.llms.moonshot import Moonshot
from langchain_community.chat_models.moonshot import MoonshotChat
# from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from graph_rag.graph import GraphBuilder
from graph_rag.rag import GraphRAG
from graph_rag.graph import Neo4jGraph
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from graph_rag.graph import Document
from langchain_community.vectorstores import Neo4jVector

# build graph  related
from llama_index.llms.ollama import Ollama
from langchain_groq import ChatGroq
from graph_rag.rag import HuggingFaceEmbeddings
from graph_rag.rag import BGEM3Reranker
from llama_index.core import Settings

from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
RUN_LOCAL = True

# online 
os.environ["NEO4J_URI"] = "neo4j+s://ce0ac75d.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LyuGyjYGDBM66DaKahd4dZNLIubyanxkaO3FLmaCF70"
os.environ["MOONSHOT_API_KEY"] = "sk-Zw5UirKJBYkwMqdtLfc47HNyrsPmIuOGfY2dX235zmBeFGJq"
os.environ["MOONSHOT_API_KEY2"] = "sk-wvLQ6IEKrOOeeUP0htYCUucYgFqt1wOhowBmVBmASHDFKSXO"
os.environ["MOONSHOT_API_KEY3"] = "sk-6XJd9jI3Ou2OE0zlnmcPew1yeh8ZXdK6qFSGUQXdEeVTUSzs"
os.environ["DASHSCOPE_API_KEY"] = "sk-9701adf350c2443680ba2486a765ba53"
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'prelude-culture-corner-panel-lava-5546'
os.environ["NEO4J_URI"] = 'bolt://localhost:7687'
os.environ["GROQ_API_KEY"] = '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'

if RUN_LOCAL:
    os.environ["NEO4J_USERNAME"] = 'neo4j'
    os.environ["NEO4J_PASSWORD"] = 'prelude-culture-corner-panel-lava-5546'
    # os.environ["NEO4J_URI"] = 'bolt://localhost:7687'
    # os.environ["NEO4J_URI"] = 'bolt://127.0.0.1:7687'
    os.environ["NEO4J_URI"] = 'bolt://0.0.0.0:7687'

class myMoonshotChat(MoonshotChat):
    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        parser = PydanticOutputParser(pydantic_object=schema)
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
    
device = 'mps'
llm = myMoonshotChat()   # llm = MoonshotChat()
llm2 =  myMoonshotChat(api_key=os.environ["MOONSHOT_API_KEY2"])
llm3 =  myMoonshotChat(api_key=os.environ["MOONSHOT_API_KEY3"])

# llm_groq = ChatGroq(temperature=0, model_name="llama3-groq-8b-8192-tool-use-preview") # model_name="mixtral-8x7b-32768")
# llm = llm_groq
# from vllm import LLM
# prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
# llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
# outputs = llm.generate(prompts)  # Generate texts from the prompts.
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
# llm = HuggingFacePipeline.from_model_id(
#     model_id="/nvme0/Qwen2-7B-Instruct/",
#     task="text-generation",
#     device=0,  # -1 for CPU
#     model_kwargs={"temperature": 0, "max_length": 512},
#     # model_kwargs={"temperature": 0, "quantization_config": quantization_config},  # add quntinize!!!
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
# )

st.set_page_config(page_title="灵起 & 瑞泰 RAG 私有化部署大模型", layout="wide")


@st.cache_resource
def load_model():
    # llm_qwen = Tongyi() 
    # llm_moon  = Moonshot()
    # llm  = ChatOpenAI(api_key=os.environ["MOONSHOT_API_KEY"],
    #                 base_url="https://api.moonshot.cn/v1", temperature=0, model="moonshot-v1-8k")
    # ollm_qwen4b = Ollama(model="qwen:4b", json_mode=True, request_timeout=60)

    # reranker
    reranker = BGEM3Reranker('/Users/zdx/llm/bge-reranker-v2-m3', device=device)
    # bge_m3emb = CustomBgeEmbedding('/Users/zdx/llm/bge-m3/') #'BAAI/bge-m3',  not work correct for encoder_documents

    # huggingface_embeddings = HuggingFaceEmbeddings('/Users/zdx/llm/bge_large_zh/')
    huggingface_embeddings = HuggingFaceEmbeddings('/Users/zdx/llm/bge-m3/', device)

    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"]
    )
    # print(graph.schema) # 打印图的结构信息 太长了
    vector_index = Neo4jVector.from_existing_graph(
        huggingface_embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        # index_name = 'embedding'
    )
    vector_index = vector_index.as_retriever(search_kwargs={'k': 6})
    return graph, reranker, huggingface_embeddings,  vector_index

llms = [llm, llm2, llm3]
graph, reranker, huggingface_embeddings,  vector_index = load_model()


# TODO  普通LLM支持结构化信息调研

Settings.embed_model = huggingface_embeddings
Settings.llm = llm

st.title("灵起 & 瑞泰 RAG")

def graph_content(progress_bar, status_text):
    """
    Entry point to generate a new graph. Will add controls to the UI 
    to perform these actions in the future
    """
    print("Building graph from content")
    # print(f"upload_file is {upload_file}, raw_docs is {raw_docs}")
    graph_builder = GraphBuilder(llm=llm)
    # TextLoader('/Users/zdx/llm/graphtest/input/三国演义白话文utf8.txt').load()

    # graph_builder.extract_wikipedia_content("Garmin_Forerunner")
    graph_builder.graph_text_documents(['/Users/zdx/llm/graphtest/input/三国演义白话文utf8.txt'])
    status_text.text("Complete Wikipedia Content")
    progress_bar.progress(3/3)

    graph_builder.index_graph()

    # print(graph_builder.graph.schema) # 打印图的结构信息

def reset_graph():
    """
    Will reset the graph by deleting all relationships and nodes
    """
    graph_builder = GraphBuilder(llm = llm)
    graph_builder.reset_graph()

def get_response(question: str) -> str:
    """
    For the given question will formulate a search query and use a custom GraphRAG retriever 
    to fetch related content from the knowledge graph. 

    Args:
        question (str): The question posed by the user for this graph RAG

    Returns:
        str: The results of the invoked graph based question
    """
    rag = GraphRAG(llm = llm, embedding_model=huggingface_embeddings, reranker=reranker, llms = llms, vector_index=vector_index)
    search_query = rag.create_search_query(st.session_state.chat_history, question)

    print(f'>> 1.1 query rewrite as : {search_query}' )

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""

    template = """使用中文、自然语言并保持简洁，且仅根据以下内容回答问题， 内容:
    {context}

    问题: {question}
    
    回答:"""
    prompt = ChatPromptTemplate.from_template(template)

    # recall_context = rag.retriever(search_query)
    recall_context = rag.retriever(question)
    def hook_retriever(search_query):
        return recall_context

    chain = (
        RunnableParallel(
            {
                # "context": lambda x: rag.retriever(search_query),
                "context": lambda x: hook_retriever(search_query),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Using invoke method to get response
    return chain.invoke({"chat_history": st.session_state.chat_history, "question": question}), recall_context

# st.session_state.chat_history = []
# response = get_response('张飞 跟曹操  干过活吗')
# print(response)

def init_ui():
    """
    Primary entry point for the app. Creates the chat interface that interacts with the LLM. 
    """

    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="你好, 我是一个智能助手. 有问题请问我！")
        ]

    user_query = st.chat_input("Ask a question....")
    if user_query is not None and user_query != "":
        response, recall_context = get_response(user_query)

         # Add the current chat to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        st.session_state.chat_history.append(recall_context)

    # Print the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        if isinstance(message, str):  # TODO ,this is bad
            with st.chat_message("recall content"):
                # print('>> message is ', message)
                for _str_ms in message.split('Unstructured data:'):
                    st.write(_str_ms)

    with st.sidebar:
        st.header("知识库管理")
        st.write("Below are options to populate and reset your graph database")

        # Create two columns for the buttons
        col1, col2 = st.columns(2)
        # Add Docs to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100

        uploaded_file = st.sidebar.file_uploader(
            "Add a Doc :page_facing_up:", type=["txt", 'pdf', 'docx'], key=st.session_state["file_uploader_key"]
        )
        rag_documents = None
        if uploaded_file is not None:
            rag_name = uploaded_file.name.split(".")[0]
            file_type = uploaded_file.type
            alert = st.sidebar.info("Processing Doc...", icon="🧠")
            # 保存文件
            save_file_name = uploaded_file.name
            file_path = os.path.join("uploads", save_file_name)
            print(f'>> the upload file type is {file_type} rag_name is {rag_name} and upfile is :{uploaded_file} {type(file_type)} save name is {file_path}')
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                st.success(f"已保存文件: {file_path}")
            # 创建一个按钮，并指定on_click回调函数
            load_button = st.sidebar.button("Load Document Content")
            if load_button:
                # build vector
                # if f"{rag_name}_uploaded" not in st.session_state:
                if 'text/plain' in file_type: # text/plain
                    # reader = myTextLoader(rag_name)#.read_from_object(uploaded_file)
                    # rag_documents: List[Document] = reader.read_from_object(uploaded_file)
                    rag_documents = TextLoader(file_path).load()
                elif 'pdf' in file_type:
                    reader = PyPDFLoader(file_path)
                    rag_documents: List[Document] = reader.load()
                elif 'word' in file_type:
                    loader = Docx2txtLoader(file_path)
                    rag_documents: List[Document] = loader.load()
                else:
                    raise TypeError('not surpport file type')

                graph_builder = GraphBuilder(llm=llm, llms=llms)
                progress_bar = st.progress(0)
                status_text = st.empty()
                text_chunks = graph_builder.chunk_document_text(rag_documents, chunk_size=800, chunk_overlap=100)
                # index = VectorStoreIndex.from_documents(text_chunks)  # optimize use GPU  # need a llm
                if text_chunks is not None:
                    graph_builder.graph_document_text(text_chunks=text_chunks, progress_bar = progress_bar)
                    status_text.text("Complete build kerword of uploaded document")
                progress_bar.progress(3/3)


            if f"{rag_name}_uploaded" not in st.session_state:
                pass
                # if 'text/plain' in file_type: # text/plain
                #     reader = TextLoader(rag_name)#.read_from_object(uploaded_file)
                #     rag_documents: List[Document] = reader.read_from_object(uploaded_file)
                # elif 'pdf' in file_type:
                #     rag_documents: List[Document] = reader.read(uploaded_file)
                # else:
                #     raise TypeError('not surpport file type')
                # if rag_documents:
                #     pass
                #     # TODO 
                #     # build Graph rag use some tool~ and save it into nef4j or just as temp vec
                # else:
                #     st.sidebar.error("Could not read Doc")
                # st.session_state[f"{rag_name}_uploaded"] = True
            alert.empty()

        with col1:
            # if st.button("Populate Graph"):
            #     progress_bar = st.progress(0)
            #     status_text = st.empty()
            #     graph_content(progress_bar, status_text)
            pass

        with col2:
            pass
            # if st.button("Reset Graph"):
            #     reset_graph()

if __name__ == "__main__":

    init_ui()
