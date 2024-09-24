# copy code from https://github.com/datawhalechina/self-llm/blob/master/models/MiniCPM/MiniCPM-2B-chat%20langchain%E6%8E%A5%E5%85%A5.md
import sys
sys.path.append('/home/dell/llm/graphrag_lang_neo4j')
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from graph_transformers.pandytic_output_parser import  myPydanticOutputParser   # used for parse Graph kg

from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from graph_rag.entities import graph_prompt_example
from langchain_core.runnables import RunnableLambda
from langchain_ollama.chat_models import ChatOllama
from graph_rag.entities import Entities, extract_entity_prompt, entity_nodes

RUN_ON_LINUX = True


def check_json(raw_input):
    print('>> the raw input of an chain is : \n', raw_input)
    if 'json' in raw_input:
        return raw_input.replace('json', '')
    return raw_input


class myChatOllama(ChatOllama):
    def mywith_structured_output(self, schema, *, include_raw: bool = False, **kwargs):
        if isinstance(schema, Entities):
            parser = PydanticOutputParser(pydantic_object=schema)
            # parser = myPydanticOutputParser(pydantic_object=schema)
            required_keys_dc = {'names':['id', 'type']}
        else:
            # parser = myPydanticOutputParser(pydantic_object=schema)
            parser = PydanticOutputParser(pydantic_object=schema)
            required_keys_dc = {  # for knowledge graph extract
                'relationships': ['source_node_id', 'source_node_type', 'target_node_id', 'target_node_type', 'type'],
                'nodes':['id', 'type']
            }

        if isinstance(parser, myPydanticOutputParser):
            parser.insert_schema(schema=required_keys_dc)
        # Prompt llm.with_structured_output(schema, include_raw=True)
        old_instruct = parser.get_format_instructions()
        exampled_instruct = graph_prompt_example + '输出schema要求如下：' +  old_instruct.split('Here is the output schema:')[1]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                    "回答用户的问题. 将输出包装为 `json` 格式\n{format_instructions}",
                ),
                ("human", "{query}"),
            ]
        ).partial(format_instructions=exampled_instruct if include_raw else '')
        # print(prompt.invoke(query).to_string())
        chain = prompt | self | RunnableLambda(check_json) | parser
        return chain


if RUN_ON_LINUX == False:
    from langchain_community.chat_models.moonshot import MoonshotChat
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
            chain = prompt | self | parser
            return chain


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
            chain = prompt | self | parser
            return chain

class MiniCPM_LLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        device = "cuda"
        # 以下是加载2B模型的code
        if model_path is not None and len(model_path) > 2:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16, device_map=device) #device_map="auto")
        else:
            path = "openbmb/MiniCPM3-4B"
            self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)


        self.model = self.model.eval()
        print(">> 完成本地模型 minicpm3-4b的加载")

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 通过模型获得输出
        responds, history = self.model.chat(self.tokenizer, prompt, temperature=0.5, top_p=0.8, repetition_penalty=1.02)
        return responds
        
    @property
    def _llm_type(self) -> str:
        return "MiniCPM_LLM"

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
        ).partial(format_instructions=parser.get_format_instructions() if not include_raw else '')
        # print(prompt.invoke(query).to_string())
        # chain = prompt | llm | parser
        chain = prompt | self | RunnableLambda(check_json) | parser
        return chain


class myLocalLM4Graph(MiniCPM_LLM):
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
        exampled_instruct = graph_prompt_example + '输出schema要求如下：' +  old_instruct.split('Here is the output schema:')[1]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                    "回答用户的问题. 将输出包装为 `json` 格式\n{format_instructions}",
                ),
                ("human", "{query}"),
            ]
        ).partial(format_instructions=exampled_instruct if include_raw else '')
        # print(prompt.invoke(query).to_string())
        chain = prompt | self | RunnableLambda(check_json) | parser
        return chain

if __name__ == "__main__":
    import os
    # os.environ['http_proxy'] = 
    os.environ['http_proxy']="http://127.0.0.1:7890"
    os.environ['https_proxy']="http://127.0.0.1:7890"
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # llm = MiniCPM_LLM('/root/autodl-tmp/OpenBMB/MiniCPM-2B-sft-fp32')
    llm = MiniCPM_LLM(None)
    res = llm('你好, 推荐5个北京的景点。')
    print('>> chat res is :')
