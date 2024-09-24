from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List

# class Entities(BaseModel):
#     """
#     Identify and capture information about entities from text
#     """

#     names: List[str] = Field(
#         description=
#             "All the objects, person, organization, authors, keywords, title, abstract, institutions, citations, references, " + 
#             "figures_tables, data, methods, results, discussion, conclusion, funding, classification_codes, proper_nouns, locations, " + 
#             "time, laws_regulations, technical_terms, products_brands, species_taxonomy, codes_algorithms, statistical_indicators, " + 
#             "research_topics, or business entities that appear in the text",
#     )
class Entities(BaseModel):
    """
    从文本中识别并捕获实体的信息
    """

    names: List[str] = Field(
        description=(
            "文本中出现的所有实体名称列表，包括但不限于："
            "物体、个人、组织、作者、关键词、标题、摘要、机构、引用、参考文献、图表、数据、"
            "方法、结果、讨论、结论、资助、分类代码、专有名词、地点、时间、法律规章、技术术语、"
            "产品品牌、物种分类、代码算法、统计指标、研究主题或商业实体。"
            "这些实体应该是人类可读的唯一标识符，并且应该与文本中的内容一致。"
        )
    )
entity_nodes = [
    '物体', '个人', '组织', '商业机构', '作者', '关键词', '标题', '摘要', '机构', '引用', '参考文献',
    '图表', '数据', '方法', '结果', '讨论', '结论', '资助', '分类代码', '专有名词', '地点', '时间',
    '法律规章', '技术术语', '产品品牌', '物种分类', '代码算法', '统计指标', '研究主题',
]

NODE_LIST = ["物体","个人","组织","作者","关键词","标题","摘要","机构","引用","参考文献","图表","数据",
            "方法","结果","讨论","结论","资助","分类代码","专有名词","地点","时间","法律规章","技术术语",
            "产品品牌","物种分类","代码算法","统计指标","研究主题","商业实体"]

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

graph_example = {
  "nodes": [
    {
      "id": "陶建华",
      "type": "个人"
    },
    {
      "id": "巫英才",
      "type": "个人"
    },
    {
      "id": "多模态人机交互综述",
      "type": "标题"
    },
    {
      "id": "第27卷/第6期/2022年6月",
      "type": "文献"
    }
  ],
  "relationships": [
    {
      "source_node_id": "陶建华",
      "source_node_type": "个人",
      "target_node_id": "多模态人机交互综述",
      "target_node_type": "标题",
      "type": "作者"
    },
    {
      "source_node_id": "巫英才",
      "source_node_type": "个人",
      "target_node_id": "第27卷/第6期/2022年6月",
      "target_node_type": "文献",
      "type": "作者"
    }
  ]
}

graph_prompt_example = """
您需要根据提供的文本提取信息，禁止翻译提取信息，并将其格式化为符合给定 JSON schema 的 JSON 实例并直接输出。
正确的格式化示例：
{'nodes': [{'id': '陶建华', 'type': '个人'},
  {'id': '巫英才', 'type': '个人'},
  {'id': '多模态人机交互综述', 'type': '标题'},
  {'id': '第27卷/第6期/2022年6月', 'type': '文献'}],
 'relationships': [{'source_node_id': '陶建华',
   'source_node_type': '个人',
   'target_node_id': '多模态人机交互综述',
   'target_node_type': '标题',
   'type': '作者'},
  {'source_node_id': '巫英才',
   'source_node_type': '个人',
   'target_node_id': '第27卷/第6期/2022年6月',
   'target_node_type': '文献',
   'type': '作者'}]}
错误的格式化，放错层级关系,示例：
 {"properties": {'nodes': [{'id': '陶建华', 'type': '个人'},
  {'id': '巫英才', 'type': '个人'},
  {'id': '多模态人机交互综述', 'type': '标题'},
  {'id': '第27卷/第6期/2022年6月', 'type': '文献'}],
 'relationships': 
  [{'source_node_id': '陶建华',
   'source_node_type': '个人',
   'target_node_id': '多模态人机交互综述',
   'target_node_type': '标题',
   'type': '作者'}]}}
缺少必要的字段，以及在输出结果前加了json:
json
{'nodes': [{'id': '陶建华', 'type': '个人'},
  {'id': '巫英才', 'type': '个人'},
  {'id': '多模态人机交互综述', 'type': '标题'},
  {'id': '第27卷/第6期/2022年6月', 'type': '文献'}],
 'relationships': [{'source_node_id': '陶建华',
   'source_node_type': '个人',
   'target_node_id': '多模态人机交互综述',
   'target_node_type': '标题',
   'type': '作者'},
  {'source_node_id': '巫英才',
   'source_node_type': '个人',
   'type': '作者'}]}
"""
scheme_prompt = """
输出的 JSON schema:
{
  "properties": {
    "nodes": {
      "title": "Nodes",
      "description": "节点列表",
      "type": "array",
      "items": {
        "$ref": "#/definitions/SimpleNode"
      }
    },
    "relationships": {
      "title": "Relationships",
      "description": "关系列表",
      "type": "array",
      "items": {
        "$ref": "#/definitions/SimpleRelationship"
      }
    }
  },
  "definitions": {
    "SimpleNode": {
      "title": "SimpleNode",
      "type": "object",
      "properties": {
        "id": {
          "title": "Id",
          "description": "名称或人类可读的唯一标识符。",
          "type": "string"
        },
        "type": {
          "title": "Type",
          "description": "节点的类型或标签。",
          "type": "string"
        }
      },
      "required": ["id", "type"]
    },
    "SimpleRelationship": {
      "title": "SimpleRelationship",
      "type": "object",
      "properties": {
        "source_node_id": {
          "title": "Source Node Id",
          "description": "源节点的名称或人类可读的唯一标识符。",
          "type": "string"
        },
        "source_node_type": {
          "title": "Source Node Type",
          "description": "源节点的类型或标签。",
          "type": "string"
        },
        "target_node_id": {
          "title": "Target Node Id",
          "description": "目标节点的名称或人类可读的唯一标识符。",
          "type": "string"
        },
        "target_node_type": {
          "title": "Target Node Type",
          "description": "目标节点的类型或标签。",
          "type": "string"
        },
        "type": {
          "title": "Type",
          "description": "关系的类型。",
          "type": "string"
        }
      },
      "required": ["source_node_id", "source_node_type", "target_node_id", "target_node_type", "type"]
    }
  }
}
"""
