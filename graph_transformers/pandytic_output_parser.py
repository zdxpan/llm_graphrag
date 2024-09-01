import json
from typing import Generic, List, Optional, Type, Dict
from pydantic import BaseModel, Field


import pydantic  # pydantic: ignore

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import (
    PYDANTIC_MAJOR_VERSION,
    PydanticBaseModel,
    TBaseModel,
)

from langchain_core.output_parsers import PydanticOutputParser

def find_required_keys(d, path=(), results=None):
    if results is None:
        results = []  # 初始化结果列表
    if isinstance(d, dict):
        for k, v in d.items():
            if k == 'required':
                # {' -> '.join(path + (k,)): v}
                results.append((' -> '.join(path + (k,)), v))  # 存储路径和值
            else:
                find_required_keys(v, path + (k,), results)
    elif isinstance(d, list):
        for index, item in enumerate(d):
            find_required_keys(item, path + (str(index),), results)
    return results


class myPydanticOutputParser(PydanticOutputParser):
    """Parse an output using a pydantic model."""

    pydantic_object: Type[TBaseModel]  # type: ignore   Dynamic GraphSchema   self.pydantic_object.schema() is dict
    required_keys_dc: Dict[str, List[str]] = Field(default={})
    """The pydantic model to parse."""

    # def __init__(self, pydantic_object: Type[TBaseModel]) -> None:
    #     """Initialize the parser."""
    #     super().__init__(pydantic_object=pydantic_object)  # 确保传递 pydantic_object 参数
    #     self.required_keys_dc = {
    #         'relationships': ['source_node_id', 'source_node_type', 'target_node_id', 'target_node_type', 'type'],
    #         'nodes': ['id', 'type']
    #     }
    
    def insert_schema(self, schema: dict) -> None:
        """Insert the schema into the parser."""
        print('>> insert an dict of requires keys in  myPydanticOutputPaser')
        self.required_keys_dc = schema

    def _parse_obj(self, obj: dict) -> TBaseModel:
        if PYDANTIC_MAJOR_VERSION == 2:
            try:
                if issubclass(self.pydantic_object, pydantic.BaseModel):
                    return self.pydantic_object.model_validate(obj)
                elif issubclass(self.pydantic_object, pydantic.v1.BaseModel):
                    root_obj = self.pydantic_object._enforce_dict_if_root(obj)
                    sch = self.pydantic_object.schema()
                    # sch['definitions']  # 'Property', 'SimpleNode', 'RelationshipProperty', 'SimpleRelationship']
                    # sch['definitions']['SimpleRelationship']['required'] # ['source_node_id', 'source_node_type', 'target_node_id', 'target_node_type', 'type']
                    # obj['relationships'][0].keys() # [' dict_keys(['source_node_id', 'source_node_type', 'target_node_id', 'target_node_type', 'type']) may missing key
                    path_required_keys = find_required_keys(sch)
                    path_required_keys = {it[0].lower(): it[1] for it in path_required_keys}
                    # 0 = ('definitions -> Property -> required', ['key', 'value'])
                    # 1 = ('definitions -> SimpleNode -> required', ['id', 'type'])
                    # 2 = ('definitions -> RelationshipProperty -> required', ['key', 'value'])
                    # 3 = ('definitions -> SimpleRelationship -> required', ['source_node_id', 'source_node_type', 'target_node_id', 'target_node_type', 'type'])
                    # only assume missing key error , not struct error
                    # 最好由外部传进来，才能更灵活定义，才不会出错，才会特别简单~
                    required_keys_dc = {
                        'relationships': ['source_node_id', 'source_node_type', 'target_node_id', 'target_node_type', 'type'],
                        'nodes':['id', 'type']
                    }
                    rebuilt_obj = {}
                    for k,v in root_obj.items():
                        required_keys = self.required_keys_dc[k.lower()]
                        rebuilt_obj[k] = []
                        if isinstance(v, dict):
                            v = [v]
                        for item in v:
                            missing_keys = [key for key in required_keys if key not in item.keys()]
                            if len(missing_keys) > 0:
                                continue
                            rebuilt_obj[k].append(item)
                    return self.pydantic_object.parse_obj(rebuilt_obj)
                else:
                    raise OutputParserException(
                        f"Unsupported model version for PydanticOutputParser: \
                            {self.pydantic_object.__class__}"
                    )
            except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
                raise self._parser_exception(e, obj)
        else:  # pydantic v1
            try:
                return self.pydantic_object.parse_obj(obj)
            except pydantic.ValidationError as e:
                raise self._parser_exception(e, obj)


