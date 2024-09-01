import os
import sys

sys.path.append('/Users/zdx/llm/graphrag_lang_neo4j')

from langchain_community.chat_models.moonshot import MoonshotChat
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
    [("system","You are extracting objects, person, organization, " +
            "or business entities from the text.",
        ),("human","Use the given format to extract information from the following "
            "input: {question}",
        ),]
)
# Neo4j Client Setup
# os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["MOONSHOT_API_KEY"] = "sk-npGQpPfAFwepce8sQnxEh2c2KJxZPb6LPSS6mvH361uMKZPu"
# online 
os.environ["NEO4J_URI"] = "neo4j+s://ce0ac75d.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "LyuGyjYGDBM66DaKahd4dZNLIubyanxkaO3FLmaCF70"
os.environ["MOONSHOT_API_KEY"] = "sk-Zw5UirKJBYkwMqdtLfc47HNyrsPmIuOGfY2dX235zmBeFGJq"
os.environ["DASHSCOPE_API_KEY"] = "sk-9701adf350c2443680ba2486a765ba53"

os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'prelude-culture-corner-panel-lava-5546'
os.environ["NEO4J_URI"] = 'bolt://localhost:7687'
os.environ["GROQ_API_KEY"] = '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'

from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_experimental.graph_transformers import LLMGraphTransformer
from graph_transformers import LLMGraphTransformer
from graph_rag.entities import graph_prompt_example

# from graph_rag.myllm import myMoonshotChat  # 不行，这个function call 没法提取结果，不适用于groq以外的其他llm


    
llm = MoonshotChat()

from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.output_parsers import PydanticOutputParser


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
mymoon_llm = myMoonshotChat()
# 中英双语提示词，使得模型更加健壮

entity_nodes = [
    '物体', '个人', '组织', '商业机构', '作者', '关键词', '标题', '摘要', '机构', '引用', '参考文献',
    '图表', '数据', '方法', '结果', '讨论', '结论', '资助', '分类代码', '专有名词', '地点', '时间',
    '法律规章', '技术术语', '产品品牌', '物种分类', '代码算法', '统计指标', '研究主题',
]
from graph_rag.entities import Entities, entity_nodes

extract_entity_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # "You are extracting objects, person, organization, " +
            # "or business entities from the text.",
            # "你正在从文本中提取物体、个人、组织、作者、关键词、标题、摘要、机构、引用、参考文献、
            # 图表、数据、方法、结果、讨论、结论、资助、分类代码、专有名词、地点、时间、法律规章、
            # 技术术语、产品品牌、物种分类、代码算法、统计指标、研究主题或商业实体。",
            f"你正在从文本中提取{'、'.join(entity_nodes)}等实体。",
        ),
        (
            "human",
            # "Use the given format to extract information from the following "
            "请使用给定的格式从以下输入中提取信息："
            "input: {question}",
        ),
    ]
)

# test_chain1 = extract_entity_prompt | mymoon_llm.with_structured_output(Entities, include_raw=False)


# entities = test_chain1.invoke({"question": '1.5.2语音情感识别\
# 语音情感识别研究的早期阶段遵循传统的模式识别流程，即先进行特征提取,然后进行分类器设计。特征提取阶段大多依赖于手工设计的与情感相关的声学特征。大体上，这些声学特征可以分为3 类,分别是韵律学特征、谱相关特征以及音质特征(Zhuge 等,2021)。开源工具 openSMILE(韩文静等,2014)通常用于提取一些经典的情感声学特征'})
# print('>> test with_structured_output to extract entitys is ', entities)


from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_community.graphs import Neo4jGraph
from graph_transformers.pandytic_output_parser   import myPydanticOutputParser

graph = Neo4jGraph(
    # url=url,
    # username=username,
    # password=password, 
    # database = database
)


class myMoonshotChat4Graph(MoonshotChat):
    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        parser = myPydanticOutputParser(pydantic_object=schema)
        required_keys_dc = {  # for knowledge graph extract
            'relationships': ['source_node_id', 'source_node_type', 'target_node_id', 'target_node_type', 'type'],
            'nodes':['id', 'type']
        }
        parser.insert_schema(schema=required_keys_dc)
        # parser = PydanticOutputParser(pydantic_object=schema)
        # Prompt llm.with_structured_output(schema, include_raw=True)
        old_instruct = parser.get_format_instructions()
        exampled_instruct = graph_prompt_example + '输出schema如下：' +  old_instruct.split('Here is the output schema:')[1]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                    "回答用户的问题. 将输出包装为 `json` 格式\n{format_instructions}",
                ),
                ("human", "{query}"),
            ]
        ).partial(format_instructions=exampled_instruct if include_raw else '')
        # print(prompt.invoke(query).to_string())
        chain = prompt | llm | parser
        return chain


device = 'mps'
llm = myMoonshotChat4Graph()

llm_transformer = LLMGraphTransformer(   # bi
    llm=llm,
    allowed_nodes = entity_nodes,         # allowed_nodes: List[str] = [],
    # allowed_relationships: List[str] = [],
    # prompt: Optional[langchain_core.prompts.chat.ChatPromptTemplate] = None,
    strict_mode=False, # strict_mode: bool = True,
    node_properties=True, # node_properties: Union[bool, List[str]] = False,
    relationship_properties = True, # relationship_properties: Union[bool, List[str]] = False,
    )

pdf_file_path = '/Users/zdx/Documents/多模态人机交互综述_陶建华.pdf'

# raw_documents = TextLoader(r"/Users/zdx/llm/graphtest/input/三国演义白话文utf8.txt").load()
raw_documents = PyPDFLoader(pdf_file_path).load()
text_splitter = TokenTextSplitter( chunk_size=800, chunk_overlap=100)
docs = text_splitter.split_documents(raw_documents)
# Chunk the document
documents = text_splitter.split_documents(raw_documents)
print('>> total docments : is ', len(documents))
# print(f"Chunk {i}: {documents[0].page_content}")

# # 打印转换过程中的中间结果
graph_documents = llm_transformer.convert_to_graph_documents(documents[2:4])  # take long time 
for i, graph_doc in enumerate(graph_documents):
    print(f"Graph Document {i}:")
    print(f"Nodes: {graph_doc.nodes}")
    print(f"Relationships: {graph_doc.relationships}")
    break

graph.add_graph_documents(
    graph_documents,
    baseEntityLabel=True,
    include_source=True
)


