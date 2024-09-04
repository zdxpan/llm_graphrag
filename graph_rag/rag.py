"""
Module to manage all GraphRAG functions. Mainly revolves around building up 
a retriever that can parse knowledge graphs and return related results
"""
import torch
from typing import List
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from graph_rag.entities import Entities
from langchain_core.documents import Document
from llama_index.core.retrievers import VectorIndexRetriever # need llm

from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from graph_rag.entities import NODE_LIST as entity_nodes

from FlagEmbedding import FlagReranker
from FlagEmbedding import BGEM3FlagModel


# 构建非结构化检索（verctor_index）
# 文本向量化 混合检索 文本和关键词检索器构建
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel

# Settings._embed_model.get_text_embedding('123')
class HuggingFaceEmbeddings():
    # huggingface_embeddings = HuggingFaceEmbeddings('/Users/zdx/llm/bge_large_zh/')
    # embed1 = huggingface_embeddings.embed_query('关于张飞')
    # print(len(embed1))
    def __init__(self, model_name, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def embed_query(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy().tolist()

    # def embedding_function(text):
        # outputs = self.model(**self.tokenizer(text, return_tensors='pt'))
        # ouputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()
        
    def embed_documents(self, documents):
        return [self.embed_query(doc) for doc in documents]
class CustomBgeEmbedding(Embeddings):

    def __init__(self, embed_model='BAAI/bge-m3', device='cpu'):
        """Initialize the BGEM3FlagModel."""
        self.device = device
        if isinstance(embed_model, str):
            self.model = BGEM3FlagModel(embed_model, use_fp16=True)
        else:
            self.model = embed_model
        self.dimensions = 1024
    @torch.no_grad()
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            embeddings = self.model.encode(texts, batch_size=12, max_length=8192,)['dense_vecs'] # make sure there return a list zhi少一个数组的列表
        print('bge_embed_documents', embeddings.shape)
        return embeddings.tolist()
        
    @torch.no_grad()
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
    
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / \
            torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BGEM3Reranker():
    def __init__(self, model_or_path, device='cpu'):
        if isinstance(model_or_path, str):
            # self.model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
            self.reranker = FlagReranker(model_or_path, use_fp16=True)
        else:
            self.reranker = model_or_path
        self.reranker.model = self.reranker.model.to(device)

    def bqe_rerank(self, query, documents):
        '''使用BAAI/bge-reranker-v2-m3 对检索结果进行重排序'''
        print(f'>> all documents {len(documents)}')
        with torch.no_grad():
            if len(documents) < 1:
                return []
            if isinstance(documents[0], Document):
                doc_texts = [doc.metadata['text'] for doc in documents]
            elif isinstance(documents[0], str):
                doc_texts = documents
            doc_pairs = [[query, i] for i in doc_texts]
            scores =  self.reranker.compute_score(doc_pairs, normalize=True)
            ranked_docs = sorted(zip(documents, scores), key=lambda x:x[1], reverse=True)
        # return  [doc for doc, _ in ranked_docs]
        return scores


class GraphRAG():
    """
    Class to encapsulate all methods required to query a graph for retrieval augmented generation
    """

    def __init__(self, llm=None, embedding_model = None, reranker=None, llms = None, vector_index=None):
        self.graph = Neo4jGraph()
        if llm is None:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        else:
            self.llm = llm
        self.embed_model = embedding_model
        self.reranker=reranker
        self.llms = llms
        self.vector_index = vector_index

    def create_entity_extract_chain(self):
        """
        Creates a chain which will extract entities from the question posed by the user. 
        This allows us to search the graph for nodes which correspond to entities more efficiently

        Returns:
            Runnable: Runnable chain which uses the LLM to extract entities from the users question
        """
        prompt = ChatPromptTemplate.from_messages(
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

        entity_extract_chain = prompt | self.llm.with_structured_output(Entities)
        # entity_extract_chain2 = prompt | self.llm.with_structured_output(Entities)
        return entity_extract_chain

    def create_entity_extract_chain_2(self):
        """
        Creates a chain which will extract entities from the question posed by the user. 
        This allows us to search the graph for nodes which correspond to entities more efficiently

        Returns:
            Runnable: Runnable chain which uses the LLM to extract entities from the users question
        """
        parser = PydanticOutputParser(pydantic_object=Entities)
        # Prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                    "回答用户的问题. 将输出包装为 `json` 格式\n{format_instructions}",
                ),
                ("human", "{question}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        # print(prompt.invoke(query).to_string())

        entity_extract_chain = prompt | self.llm | parser
        # entity_extract_chain.invoke({"question": query})
        return entity_extract_chain

    def generate_full_text_query(self, input_query: str) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspellings.

        Args:
            input_query (str): The extracted entity name pulled from the users question

        Returns:
            str: _description_
        """
        full_text_query = ""

        # split out words and remove any special characters reserved for cipher query
        words = [el for el in remove_lucene_chars(input_query).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def structured_retriever(self, question: str) -> str:
        """
        Creates a retriever which will use entities extracted from the users query to 
        request context from the Graph and return the neighboring nodes and edges related
        to that query. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The fully formed Graph Query which will retrieve the 
                 context relevant to the users question
        """

        entity_extract_chain = self.create_entity_extract_chain()
        # entity_extract_chain = self.create_entity_extract_chain_2()
        
        result = ""
        entities = entity_extract_chain.invoke({"question": question})
        print('>> 3  entities from create_entity_extract_chain ', entities)
        if entities is None:
            return result
        for idx, entity in enumerate(entities.names):
            # filter some entitys
            response = self.graph.query(
                # """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                """CALL db.index.fulltext.queryNodes('keyword', $query, {limit:2})
                YIELD node,score
                CALL {
                WITH node
                MATCH (node)-[r]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 20
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
            print(f'>> according {entities} query from graph get {result}')
        return result


    def create_vector_index(self) -> Neo4jVector:
        """
        Uses the existing graph to create a vector index. This vector representation
        is based off the properties specified. Using OpenAIEmbeddings since we are using 
        GPT-4o as the model. 

        Returns:
            Neo4jVector: The vector representation of the graph nodes specified in the configuration
        """
        if self.vector_index is None:
            vector_index = Neo4jVector.from_existing_graph(
                OpenAIEmbeddings() if self.embed_model is None else self.embed_model,
                search_type="hybrid",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding",
                # index_name = 'embedding'
            )
            self.vector_index = vector_index
        return self.vector_index


    def retriever(self, question: str) -> str:
        """
        The graph RAG retriever which combines both structured and unstructured methods of retrieval 
        into a single retriever based off the users question. 
        
        Args:
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The retrieved data from the graph in both forms
        """
        print(f">> 1.2 Search query: {question}")
        vector_index = self.create_vector_index()
        # vector_index_retriever = self.vector_retriever(top_k=10)
        # retriver_res = vector_index_retriever.retrieve(question)
        # unstructured_data = [el.page_content for el in vector_index.similarity_search(question, k=5)]
        unstructured_data = [el.page_content for el in vector_index.invoke(question)]
        if self.reranker is not None:
            unstructured_data_score = self.reranker.bqe_rerank(question, unstructured_data)
            unstructured_data = sorted(list(zip(unstructured_data, unstructured_data_score)), key=lambda x: x[1], reverse=True)
            unstructured_data = [it[0] for it in unstructured_data]
            unstructured_data = unstructured_data[:6]  # toomuch
        print(f'>> 2 query from vector_index get {len(unstructured_data)} ')
        structured_data = self.structured_retriever(question)
        final_data = f"""Structured data:
            {structured_data}
            Unstructured data:
            {"#Document ". join(unstructured_data)}
        """
        return final_data

    def create_search_query(self, chat_history: List, question: str) -> str:
        """
        Combines chat history along with the current question into a prompt that 
        can be executed by the LLM to answer the new question with history.

        Args:
            chat_history (List): List of messages captured during this conversation
            question (str): The question posed by the user for this graph RAG

        Returns:
            str: The formatted prompt that can be sent to the LLM with question & chat history
        """
        search_query = ChatPromptTemplate.from_messages([
            (
                # "system",
                # """Given the following conversation and a follow up question, rephrase the follow 
                # up question to be a standalone question, in its original language.
                # Chat History:
                # {chat_history}
                # Follow Up Input: {question}
                # Standalone question:"""
                "system",
                """根据以下对话和后续问题，将后续问题重新表述为一个独立的、使用原始语言的问题。
                聊天记录：
                {chat_history}
                后续问题: {question}
                独立问题:"""
            )
        ])
        formatted_query = search_query.format(
            chat_history=chat_history, question=question)
        return formatted_query
