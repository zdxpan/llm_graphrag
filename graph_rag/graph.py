"""
Script responsible for build a knowledge graph using
Neo4j from unstructured text
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Iterator, Optional, Union
import tqdm
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader, TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_openai import ChatOpenAI
# 嵌入来根据其余弦距离找到相似的潜在候选者。我们将使用Graph Data Science (GDS)库中可用的图算法
from graphdatascience import GraphDataScience

import time
# project graph
logger = logging.getLogger(__name__)

# RecursiveCharacterTextSplitter 将其分成更小的文本片段。该拆分器确保每个块最大化为 200 个标记，其中重叠 20 个标记，遵守嵌入模型的上下文窗口限制，并确保不会丢失上下文的连续性。

# os.environ["NEO4J_URI"] = "bolt://localhost:7687"
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "buildkg123!"

# gds = GraphDataScience(
#     os.environ["NEO4J_URI"],
#     auth=(os.environ["NEO4J_USERNAME"], 
#           os.environ["NEO4J_PASSWORD"])
# )

# https://github.com/rahulnyk/graph_maker  #这里有一个知识图谱提取工具

class GraphBuilder():
    """
    Encapsulates the core functionality requires to build a full knowledge graph 
    from multiple sources of unstructured text

    _extended_summary_
    """
    def __init__(self, llm=None, token_len_func = None, llms = None):
        self.graph = Neo4jGraph()
        if llm is None:
            self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        else:
            self.llm = llm
        self.llms = llms
        
        self.token_len_func = token_len_func

    def chunk_document_text(self, raw_docs, chunk_size=512, chunk_overlap=24):
        """
        Accepts raw text context extracted from source and applies a chunking 
        algorithm to it. 

        Args:
            raw_docs (str): The raw content extracted from the source

        Returns:
            List: List of document chunks
        """
        # text_splitter = RecursiveCharacterTextSplitter(
        #         chunk_size = 200,
        #         chunk_overlap  = 20,
        #         length_function = bert_len,
        #         separators=['\n\n', '\n', ' ', ''],
        #     )
        # documents = text_splitter.create_documents(all_text_docs)
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(raw_docs)
        return docs

    def graph_document_text(self, text_chunks, save_dir = None, progress_bar=None):
        """
        Uses experimental LLMGraphTransformer to convert unstructured text into a knowledge graph

        Args:
            text_chunks (List): List of document chunks
        """
        llm_transformer = LLMGraphTransformer(llm=self.llm, 
                            # allowed_nodes=["Person", "Organization", "Company", "Award", "Product", "Characteristic"],
                            # allowed_relationships=["WORKS_FOR", "HAS_AWARD", "PRODUCED_BY", "HAS_CHARACTERISTIC"]
                                            )
        llm_transformer_list = [LLMGraphTransformer(llm=lm_ ) for lm_ in self.llms]
        total_num = len(text_chunks)
        graph_documents = []
        print(f'>>> convert_to_graph_documents had {total_num}')
        def process_text(llm_transformer_, doc_item) -> List[GraphDocument]:
            # doc = Document(page_content=text)
            try:
                graph_doc = llm_transformer_.convert_to_graph_documents([doc_item])
            except Exception as e:
                time.sleep(5)
                return []
            return graph_doc
        MAX_WORKERS = len(llm_transformer_list)
        progres_inx = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submitting all tasks and creating a list of future objects
            futures = [
                executor.submit(process_text, llm_transformer_list[i % MAX_WORKERS], text_chunks[i])
                for i in range(total_num)
            ]
            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                sub_graph_document = future.result()
                graph_documents.extend(sub_graph_document)
                if progress_bar is not  None:
                    progress_bar.progress(progres_inx/total_num)
                progres_inx += 1
        print(f'>> had {len(graph_documents)} to add in to grap databases')
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )

    def chunk_and_graph(self, raw_docs):
        """
        Breaks the raw text into chunks and converts into a knowledge graph

        Args:
            raw_docs (str): The raw content extracted from the source
        """
        text_chunks = self.chunk_document_text(raw_docs)
        if text_chunks is not None:
            self.graph_document_text(text_chunks)

    def graph_text_content(self, path):
        """
        Provided with a text document, will extract and chunk the text
        before generating a graph

        Args:
            path (str): Text document path
        """
        text_docs = TextLoader(path).load()
        # print(text_docs[0])
        self.chunk_and_graph(text_docs)

    def graph_text_documents(self, paths):
        """
        Provided with an array of text documents will extract and
        graph each of them

        Args:
            paths (List): Document paths to extract and graph
        """
        all_text_docs = []
        for path in paths:
            text_docs = TextLoader(path).load()
            all_text_docs.extend(text_docs)

        text_chunks = self.chunk_document_text(all_text_docs, chunk_size=800, chunk_overlap=100)

        self.graph_document_text(text_chunks, save_dir = None)

    def index_graph(self):
        """
        Creates an index on the populated graph tp assist with efficient searches
        """
        self.graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    def reset_graph(self):
        """
        WARNING: Will clear entire graph, use with caution
        """
        self.graph.query(
            """
            MATCH (n)
            DETACH DELETE n
            """
        )
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_community.document_loaders import TextLoader, PyPDFLoader

class myTextLoader(TextLoader):
    def read_from_object(self, file_object) -> Iterator[Document]:
        """Load from a file object."""
        text = ""
        try:
            text = file_object.read()
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = detect_file_encodings(file_object)
                for encoding in detected_encodings:
                    logger.debug(f"Trying encoding: {encoding.encoding}")
                    try:
                        text = file_object.read().decode(encoding.encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError(f"Error loading file object") from e
        except Exception as e:
            raise RuntimeError(f"Error loading file object") from e
        
        metadata = {"source": str(self.file_path)}
        yield Document(page_content=text, metadata=metadata)



