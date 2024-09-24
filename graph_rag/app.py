"""
Primary entry point for the RAG app. Integrates other RAG functionality into a UI
sudo mkdir -p /etc/systemd/system/ollama.service.d; 
sudo nano /etc/systemd/system/ollama.service.d/override.conf

"""
import os
import sys


script_path = os.path.abspath(__file__)
parent_path = os.path.dirname(script_path)
parent_path = os.path.dirname(parent_path)

if parent_path not in sys.path:
    sys.path.append(parent_path)
# import dotenv
# dotenv.load_dotenv(dotenv_path = '../.env')

import streamlit as st
from typing import List, Iterator, Optional, Union
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
# from llama_index.llms.ollama import Ollama
from langchain_ollama.chat_models import ChatOllama
from graph_rag.myllm import myChatOllama
# from langchain_groq import ChatGroq
from graph_rag.rag import HuggingFaceEmbeddings
from graph_rag.rag import BGEM3Reranker

from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from graph_transformers.pandytic_output_parser import  myPydanticOutputParser   # used for parse Graph kg
from langchain_core.prompts import ChatPromptTemplate

from graph_rag.entities import Entities, extract_entity_prompt, entity_nodes

# Neo4j Client Setup
# os.environ["OPENAI_API_KEY"] = "sk-"
RUN_LOCAL = True
LOCAL_LLM = True
RUN_LUNX = True  # run on mac

# online 
os.environ["NEO4J_URI"] = "neo4j+s://ce0ac75d.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["MOONSHOT_API_KEY"] = "sk-"
os.environ["DASHSCOPE_API_KEY"] = "sk-"
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'prelude-culture-corner-panel-lava-5546'
os.environ["NEO4J_URI"] = 'bolt://localhost:7687'


if RUN_LUNX:
    device = 'cuda'
    LOCAL_RANKER_MODEL =  '/home/dell/llm/bge-reranker-v2-m3' # 'BAAI/bge-reranker-v2-m3'
    LOCAL_EMBED_MODEL = '/home/dell/llm/BAAI--bge-m3'         #  BAAI/bge-m3
else:
    device = 'mps'
    LOCAL_RANKER_MODEL =  '/Users/zdx/llm/bge-reranker-v2-m3' # 'BAAI/bge-reranker-v2-m3'
    LOCAL_EMBED_MODEL = '/Users/zdx/llm/bge-m3/'


if RUN_LOCAL:
    os.environ["NEO4J_USERNAME"] = 'neo4j'
    os.environ["NEO4J_PASSWORD"] = 'prelude-culture-corner-panel-lava-5546'
    # os.environ["NEO4J_URI"] = 'bolt://localhost:7687'
    # os.environ["NEO4J_URI"] = 'bolt://127.0.0.1:7687'
    os.environ["NEO4J_URI"] = 'bolt://0.0.0.0:7687'


from graph_rag.entities import graph_prompt_example


if RUN_LUNX:
    ollm_qwen4b = ChatOllama(model="majx13/test", json_mode=True, request_timeout=60)  # this surpprot tool usage
    ollm_extractor =  myChatOllama(model="majx13/test", json_mode=True, request_timeout=60)
    llm = ollm_qwen4b
    llms = [ollm_extractor]
else:
    from langchain_community.chat_models.moonshot import MoonshotChat
    from graph_rag.myllm import     myMoonshotChat, myMoonshotChat4Graph
    mymoon_llm = myMoonshotChat()
    llm = myMoonshotChat()   # llm = MoonshotChat() llm = MoonshotChat(client="moonshot", api_key=os.environ["MOONSHOT_API_KEY"])
    llm1 = myMoonshotChat4Graph()
    llm2 =  myMoonshotChat4Graph(api_key=os.environ["MOONSHOT_API_KEY2"])
    llm3 =  myMoonshotChat4Graph(api_key=os.environ["MOONSHOT_API_KEY3"])
    llms = [llm1, llm2, llm3]

st.set_page_config(page_title="zdx私有化部署大模型", layout="wide")


graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)
@st.cache_resource
def load_model():
    # llm_qwen = Tongyi() 
    # llm_moon  = Moonshot()
    # llm  = ChatOpenAI(api_key=os.environ["MOONSHOT_API_KEY"],
    #                 base_url="https://api.moonshot.cn/v1", temperature=0, model="moonshot-v1-8k")
    # ollm_qwen4b = Ollama(model="qwen:4b", json_mode=True, request_timeout=60)

    # reranker
    reranker = BGEM3Reranker(LOCAL_RANKER_MODEL, device=device)  # 'BAAI/bge-reranker-v2-m3', device=device) 
    # bge_m3emb = CustomBgeEmbedding('/Users/zdx/llm/bge-m3/') #'BAAI/bge-m3',  not work correct for encoder_documents

    huggingface_embeddings = HuggingFaceEmbeddings(LOCAL_EMBED_MODEL)
    # huggingface_embeddings = HuggingFaceEmbeddings('BAAI/bge-m3', device = device)  # '/Users/zdx/llm/bge-m3/', device)

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


graph, reranker, huggingface_embeddings,  vector_index = load_model()



st.title("zdx RAG V 2.0")

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

def get_response(question: str, option:str) -> str:
    """
    For the given question will formulate a search query and use a custom GraphRAG retriever 
    to fetch related content from the knowledge graph. 

    Args:
        question (str): The question posed by the user for this graph RAG

    Returns:
        str: The results of the invoked graph based question
    """
    print(">> 1.0 option is : ", option)
    rag = GraphRAG(llm = llm, embedding_model=huggingface_embeddings, reranker=reranker, llms = [], vector_index=vector_index)
    search_query = rag.create_search_query(st.session_state.chat_history[-5:], question)

    print(f'>> 1.1 query rewrite as : {search_query}' )

    # template = """使用自然语言简洁回答，并且只能根据以下内容回答问题，引用内容并总结回答， 内容:
    template = """只根据以下内容回答问题，引用内容并总结回答问题， 内容:
    {context}

    问题: {question}
    
    回答:"""
    prompt = ChatPromptTemplate.from_template(template)

    # recall_context = rag.retriever(search_query)
    recall_context = rag.retriever(question) if option == '本地知识库助手' else ''
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
    if option != '本地知识库助手':
        prompt = ChatPromptTemplate.from_template('根据用户问题进行回答:{question}')
        chain = (
            prompt
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

    with st.sidebar:
        st.header("模式")
        option = st.selectbox(
            "AI助手模式",
            ("本地知识库助手", "闲聊"),
        )

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
        local_files = os.listdir("uploads")
        local_files = [i for i in local_files if '.' != i[0]]
        st.warning(f"已上传知识 列表: {local_files}")
        if uploaded_file is not None:
            rag_name = uploaded_file.name.split(".")[0]
            file_type = uploaded_file.type
            alert = st.sidebar.info("Processing Doc...", icon="🧠")
            # 保存文件
            save_file_name = uploaded_file.name
            file_path = os.path.join("uploads", save_file_name)
            print(f'>> the upload file type is {file_type} rag_name is {rag_name} and upfile is :{uploaded_file} {type(file_type)} save name is {file_path}')
            if os.path.exists(file_path):
                st.warning(f"文件已存在，请勿重复上传！！！")
            else:
                load_button = st.sidebar.button("Load Document Content")
                if load_button:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        st.success(f"已保存文件: {file_path}")
                    # 创建一个按钮，并指定on_click回调函数
                    # build vector
                    # if f"{rag_name}_uploaded" not in st.session_state:
                    if 'text/plain' in file_type: # text/plain
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
                    
                    raw_doc  = [doc.page_content for doc in rag_documents]
                    raw_doc = '/n'.join(raw_doc)
                    st.warning(f"读取文档实际读取片段数量: {len(raw_doc)}")
                    if len(raw_doc) < 800:
                        st.error(f"文档解析失败，不支持的文档类型")
                        os.remove(file_path)

                    graph_builder = GraphBuilder(llm=llm, llms=llms)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    text_chunks = graph_builder.chunk_document_text(rag_documents, chunk_size=700, chunk_overlap=80)
                    # index = VectorStoreIndex.from_documents(text_chunks)  # optimize use GPU  # need a llm
                    if text_chunks is not None:
                        graph_builder.graph_document_text(text_chunks=text_chunks, progress_bar = progress_bar, info_writer = status_text)
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

        # with col2:
        user_input = st.text_input("确认重置知识库: yes123 or n:", "")
        if st.button("Reset Graph") and user_input == "yes123":
            print('>> reset_graph databases!!!')
            reset_graph()

    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="你好, 我是一个智能助手. 有问题请问我！")
        ]

    user_query = st.chat_input("Ask a question....")
    recall_context = ''
    if user_query is not None and user_query != "":
        print('>> history len is :', len(st.session_state.chat_history))
        response, recall_context = get_response(user_query, option)

         # Add the current chat to the chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        # st.session_state.chat_history.append(recall_context)

    # Print the chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
    if isinstance(recall_context, str) and len(recall_context) > 10:  # TODO ,this is bad
        with st.chat_message("Recall content"):
            # print('>> message is ', message)
            for _str_ms in recall_context.split('Unstructured data:'):
                
                st.write(_str_ms)
        recall_context = ''


if __name__ == "__main__":

    init_ui()
