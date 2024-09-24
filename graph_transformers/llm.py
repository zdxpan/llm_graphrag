import asyncio
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast
import copy
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.runnables import RunnableConfig

examples = [
    # {
    #     "text": (
    #         "Adam is a software engineer in Microsoft since 2009, "
    #         "and last year he got an award as the Best Talent"
    #     ),
    #     "head": "Adam",
    #     "head_type": "Person",
    #     "relation": "WORKS_FOR",
    #     "tail": "Microsoft",
    #     "tail_type": "Company",
    # },
    # {
    #     "text": (
    #         "Adam is a software engineer in Microsoft since 2009, "
    #         "and last year he got an award as the Best Talent"
    #     ),
    #     "head": "Adam",
    #     "head_type": "Person",
    #     "relation": "HAS_AWARD",
    #     "tail": "Best Talent",
    #     "tail_type": "Award",
    # },
    # {
    #     "text": (
    #         "Microsoft is a tech company that provide "
    #         "several products such as Microsoft Word"
    #     ),
    #     "head": "Microsoft Word",
    #     "head_type": "Product",
    #     "relation": "PRODUCED_BY",
    #     "tail": "Microsoft",
    #     "tail_type": "Company",
    # },
    # {
    #     "text": "Microsoft Word is a lightweight app that accessible offline",
    #     "head": "Microsoft Word",
    #     "head_type": "Product",
    #     "relation": "HAS_CHARACTERISTIC",
    #     "tail": "lightweight app",
    #     "tail_type": "Characteristic",
    # },
    # {
    #     "text": "Microsoft Word is a lightweight app that accessible offline",
    #     "head": "Microsoft Word",
    #     "head_type": "Product",
    #     "relation": "HAS_CHARACTERISTIC",
    #     "tail": "accessible offline",
    #     "tail_type": "Characteristic",
    # },
    {
        "text": "亚当自2009年以来就是微软的软件工程师，去年他获得了最佳人才奖。",
        "head": "亚当",
        "head_type": "人物",
        "relation": "工作于",
        "tail": "微软",
        "tail_type": "公司",
    },
    {
        "text": "亚当自2009年以来就是微软的软件工程师，去年他获得了最佳人才奖。",
        "head": "亚当",
        "head_type": "人物",
        "relation": "获得奖项",
        "tail": "最佳人才",
        "tail_type": "奖项",
    },
    {
        "text": "微软是一家科技公司，提供多种产品，如Microsoft Word。",
        "head": "Microsoft Word",
        "head_type": "产品",
        "relation": "由...生产",
        "tail": "微软",
        "tail_type": "公司",
    },
    {
        "text": "Microsoft Word是一个轻量级应用，可以离线使用。",
        "head": "Microsoft Word",
        "head_type": "产品",
        "relation": "具有特性",
        "tail": "轻量级应用",
        "tail_type": "特性",
    },
    {
        "text": "Microsoft Word是一个轻量级应用，可以离线使用。",
        "head": "Microsoft Word",
        "head_type": "产品",
        "relation": "具有特性",
        "tail": "可以离线使用",
        "tail_type": "特性",
    },
]

system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Try to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)
system_prompt = (
    "# 知识图谱构建指南\n"
    "## 1. 简介\n"
    "你是一个高级算法，专为从文本中提取信息并以结构化格式构建知识图谱而设计。尽可能准确地从文本中提取信息，不要添加文本中未明确提及的信息，保持原始信息原样禁止翻译。\n"
    "- **实体** 代表实体和概念。\n"
    "- 目标是使知识图谱简单明了，便于广大用户理解。\n"
    "## 2. 标记实体\n"
    "- **一致性**：确保使用可用的类型作为实体标签。使用基础或通用类型作为实体标签。\n"
    "- 例如，当你识别出一个代表个人的实体时，总是将其标记为 **人**。避免使用更具体的术语如 数学家 或科学家。\n"
    "- **实体ID**：不要使用整数作为实体ID。实体ID应该是文本中找到的名称或人类可读的标识符。\n"
    "- **关系**代表实体或概念之间的连接。\n"
    "在构建知识图谱时，确保关系类型的一致性和普遍性。避免使用特定和临时的类型，如成为教授，\n"
    "而应使用更普遍和持久的关系类型，如教授。确保使用普遍和持久的关系类型！\n"
    "## 3. 指代消解\n"
    "- **保持实体一致性**：在提取实体时，确保一致性至关重要。\n"
    "如果一个实体，如约翰·多伊，在文本中多次提及但被不同的名称或代词例如约翰，他所指代，\n"
    "在知识图谱中始终使用该实体的最完整标识符。在这个例子中，使用约翰·多伊作为实体ID。\n"
    "记住，知识图谱应该是连贯且易于理解的，因此保持实体引用的一致性至关重要。\n"
    "## 4. 严格遵守\n"
    "严格遵守规则和给定的scheme 格式不缺少字段。不遵守将导致提取知识图谱过程被终止。"
)

default_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            # (
            #     "Tip: Make sure to answer in the correct format and do "
            #     "not include any explanations. "
            #     "Use the given format to extract information from the "
            #     "following input: {input}"
            # ),
            (
                "提示：确保按照正确的格式回答，不要包含任何解释。"
                "使用给定的格式从以下输入中提取信息：{input}"
            ),
        ),
    ]
)


def _get_additional_info(input_type: str) -> str:
    # Check if the input_type is one of the allowed values
    if input_type not in ["node", "relationship", "property"]:
        raise ValueError("input_type must be 'node', 'relationship', or 'property'")

    # Perform actions based on the input_type
    if input_type == "node":
        return (
            # "Ensure you use basic or elementary types for node labels.\n"
            # "For example, when you identify an entity representing a person, "
            # "always label it as **'Person'**. Avoid using more specific terms "
            # "like 'Mathematician' or 'Scientist'"
            "确保你使用基础或初级类型作为节点标签。\n"
            "例如，当你识别出一个代表个人的实体时，总是将其标记为 **人**。避免使用更具体的术语"
            "如数学家或科学家"
        )
    elif input_type == "relationship":
        return (
            # "Instead of using specific and momentary types such as "
            # "'BECAME_PROFESSOR', use more general and timeless relationship types "
            # "like 'PROFESSOR'. However, do not sacrifice any accuracy for generality"
            "不要使用特定和临时的类型，如成为教授，而应使用更普遍和持久的关系类型，如教授。不要为了普遍性而牺牲准确性"
        )
    elif input_type == "property":
        return ""
    return ""


def optional_enum_field(
    enum_values: Optional[List[str]] = None,
    description: str = "",
    input_type: str = "node",
    llm_type: Optional[str] = None,
    **field_kwargs: Any,
) -> Any:
    """Utility function to conditionally create a field with an enum constraint."""
    # Only openai supports enum param
    if enum_values and llm_type == "openai-chat":
        return Field(
            ...,
            enum=enum_values,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    elif enum_values:
        return Field(
            ...,
            description=f"{description}. Available options are {enum_values}",
            **field_kwargs,
        )
    else:
        additional_info = _get_additional_info(input_type)
        return Field(..., description=description + additional_info, **field_kwargs)


class _Graph(BaseModel):
    nodes: Optional[List]
    relationships: Optional[List]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            # "extracted head entity like Microsoft, Apple, John. "
            # "Must use human-readable unique identifier."
            "提取的头部实体，例如微软、苹果、约翰。必须使用人类可读的唯一标识符。"
        )
    )
    head_type: str = Field(
        # description="type of the extracted head entity like Person, Company, etc"
        description="头部实体的类型，例如个人、公司等。"
    )
    relation: str = Field(
        # description="relation between the head and the tail entities"
        description="头部实体和尾部实体之间的关系。"
    )
    tail: str = Field(
        description=(
            # "extracted tail entity like Microsoft, Apple, John. "
            # "Must use human-readable unique identifier."
            "提取的尾部实体，例如微软、苹果、约翰。"
            "必须使用人类可读的唯一标识符。"
        )
    )
    tail_type: str = Field(
        # description="type of the extracted tail entity like Person, Company, etc"
        description="提取的尾部实体的类型，例如个人、公司等。"
    )


def create_unstructured_prompt(
    node_labels: Optional[List[str]] = None, rel_types: Optional[List[str]] = None
) -> ChatPromptTemplate:
    node_labels_str = str(node_labels) if node_labels else ""
    rel_types_str = str(rel_types) if rel_types else ""
    base_string_parts = [
        "You are a top-tier algorithm designed for extracting information in "
        "structured formats to build a knowledge graph. Your task is to identify "
        "the entities and relations requested with the user prompt from a given "
        "text. You must generate the output in a JSON format containing a list "
        'with JSON objects. Each object should have the keys: "head", '
        '"head_type", "relation", "tail", and "tail_type". The "head" '
        "key must contain the text of the extracted entity with one of the types "
        "from the provided list in the user prompt.",
        f'The "head_type" key must contain the type of the extracted head entity, '
        f"which must be one of the types from {node_labels_str}."
        if node_labels
        else "",
        f'The "relation" key must contain the type of relation between the "head" '
        f'and the "tail", which must be one of the relations from {rel_types_str}.'
        if rel_types
        else "",
        f'The "tail" key must represent the text of an extracted entity which is '
        f'the tail of the relation, and the "tail_type" key must contain the type '
        f"of the tail entity from {node_labels_str}."
        if node_labels
        else "",
        "Attempt to extract as many entities and relations as you can. Maintain "
        "Entity Consistency: When extracting entities, it's vital to ensure "
        'consistency. If an entity, such as "John Doe", is mentioned multiple '
        "times in the text but is referred to by different names or pronouns "
        '(e.g., "Joe", "he"), always use the most complete identifier for '
        "that entity. The knowledge graph should be coherent and easily "
        "understandable, so maintaining consistency in entity references is "
        "crucial.",
        "IMPORTANT NOTES:\n- Don't add any explanation and text.",
    ]
    base_string_parts = [
        "你是一个高级算法，专为从文本中提取信息并以结构化格式构建知识图谱而设计。你的任务是根据用户提示从给定文本中识别实体和关系。你必须以JSON格式生成输出，包含一个列表，其中每个对象应有以下键：'head', 'head_type', 'relation', 'tail', 和 'tail_type'。'head' 键必须包含提取的实体文本，并且类型必须是用户提供的列表中的一种。",
        f'"head_type" 键必须包含提取的头部实体的类型，该类型必须是 {node_labels_str} 中的一种。' if node_labels else "",
        f'"relation" 键必须包含 "head" 和 "tail" 之间的关系类型，该类型必须是 {rel_types_str} 中的一种。' if rel_types else "",
        f'"tail" 键必须表示关系中的尾部实体的文本，而 "tail_type" 键必须包含尾部实体的类型，该类型必须是 {node_labels_str} 中的一种。' if node_labels else "",
        "尽可能提取尽可能多的实体和关系。保持实体一致性：在提取实体时，确保一致性至关重要。如果一个实体，如 '约翰·多伊'，在文本中多次提及，但被不同的名称或代词（例如 '乔'，'他'）所指代，始终使用该实体的最完整标识符。知识图谱应该是连贯且易于理解的，因此保持实体引用的一致性至关重要。",
        "重要提示：\n- 不要添加任何解释和文本，保持提取的原始信息不要翻译。",
    ]    
    system_prompt = "\n".join(filter(None, base_string_parts))

    system_message = SystemMessage(content=system_prompt)
    parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

    human_string_parts = [
        "Based on the following example, extract entities and "
        "relations from the provided text.\n\n",
        "Use the following entity types, don't use other entity "
        "that is not defined below:"
        "# ENTITY TYPES:"
        "{node_labels}"
        if node_labels
        else "",
        "Use the following relation types, don't use other relation "
        "that is not defined below:"
        "# RELATION TYPES:"
        "{rel_types}"
        if rel_types
        else "",
        "Below are a number of examples of text and their extracted "
        "entities and relationships."
        "{examples}\n"
        "For the following text, extract entities and relations as "
        "in the provided example."
        "{format_instructions}\nText: {input}",
    ]
    human_string_parts = [
        "根据以下示例，从提供的文本中提取实体和关系。\n",
        "使用以下实体类型，不要使用未在下方定义的其他实体："
        "# 实体类型："
        "{node_labels}" if node_labels else "",
        "使用以下关系类型，不要使用未在下方定义的其他关系："
        "# 关系类型："
        "{rel_types}" if rel_types else "",
        "以下是一些文本及其提取的实体和关系的示例。"
        "{examples}\n"
        "对于以下文本，请按照提供的示例提取实体和关系。"
        "{format_instructions}\n文本：{input}"
    ]
    human_prompt_string = "\n".join(filter(None, human_string_parts))
    human_prompt = PromptTemplate(
        template=human_prompt_string,
        input_variables=["input"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "node_labels": node_labels,
            "rel_types": rel_types,
            "examples": examples,
        },
    )

    human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message_prompt]
    )
    return chat_prompt


def create_simple_model(
    node_labels: Optional[List[str]] = None,
    rel_types: Optional[List[str]] = None,
    node_properties: Union[bool, List[str]] = False,
    llm_type: Optional[str] = None,
    relationship_properties: Union[bool, List[str]] = False,
) -> Type[_Graph]:
    """
    Create a simple graph model with optional constraints on node
    and relationship types.

    Args:
        node_labels (Optional[List[str]]): Specifies the allowed node types.
            Defaults to None, allowing all node types.
        rel_types (Optional[List[str]]): Specifies the allowed relationship types.
            Defaults to None, allowing all relationship types.
        node_properties (Union[bool, List[str]]): Specifies if node properties should
            be included. If a list is provided, only properties with keys in the list
            will be included. If True, all properties are included. Defaults to False.
        relationship_properties (Union[bool, List[str]]): Specifies if relationship
            properties should be included. If a list is provided, only properties with
            keys in the list will be included. If True, all properties are included.
            Defaults to False.
        llm_type (Optional[str]): The type of the language model. Defaults to None.
            Only openai supports enum param: openai-chat.

    Returns:
        Type[_Graph]: A graph model with the specified constraints.

    Raises:
        ValueError: If 'id' is included in the node or relationship properties list.
    """

    node_fields: Dict[str, Tuple[Any, Any]] = {
        "id": (
            str,
            # Field(..., description="Name or human-readable unique identifier."),
            Field(..., description="名称或人类可读的唯一标识符。"),
        ),
        "type": (
            str,
            optional_enum_field(
                node_labels,
                # description="The type or label of the node.",
                description="节点的类型或标签。",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
    }

    if node_properties:
        if isinstance(node_properties, list) and "id" in node_properties:
            # raise ValueError("The node property 'id' is reserved and cannot be used.")
            raise ValueError("保留的节点属性 'id' 不能使用。")
        # Map True to empty array
        node_properties_mapped: List[str] = (
            [] if node_properties is True else node_properties
        )

        class Property(BaseModel):
            """A single property consisting of key and value由键和值组成的单个属性"""

            key: str = optional_enum_field(
                node_properties_mapped,
                # description="Property key.",
                description="属性键。",
                input_type="property",
                llm_type=llm_type,
            )
            # value: str = Field(..., description="value")
            value: str = Field(..., description="属性值")

        node_fields["properties"] = (
            Optional[List[Property]],
            # Field(None, description="List of node properties"),
            Field(None, description="节点属性列表。"),
        )
    SimpleNode = create_model("SimpleNode", **node_fields)  # type: ignore

    relationship_fields: Dict[str, Tuple[Any, Any]] = {
        "source_node_id": (
            str,
            Field(
                ...,
                # description="Name or human-readable unique identifier of source node",
                description="源节点的名称或人类可读的唯一标识符。"
            ),
        ),
        "source_node_type": (
            str,
            optional_enum_field(
                node_labels,
                # description="The type or label of the source node.",
                description="源节点的类型或标签。",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "target_node_id": (
            str,
            Field(
                ...,
                # description="Name or human-readable unique identifier of target node",
                description="目标节点的名称或人类可读的唯一标识符。",
            ),
        ),
        "target_node_type": (
            str,
            optional_enum_field(
                node_labels,
                # description="The type or label of the target node.",
                description="目标节点的类型或标签。",
                input_type="node",
                llm_type=llm_type,
            ),
        ),
        "type": (
            str,
            optional_enum_field(
                rel_types,
                # description="The type of the relationship.",
                description="关系的类型。",
                input_type="relationship",
                llm_type=llm_type,
            ),
        ),
    }
    if relationship_properties:
        if (
            isinstance(relationship_properties, list)
            and "id" in relationship_properties
        ):
            raise ValueError(
                # "The relationship property 'id' is reserved and cannot be used."
                "保留的关系属性 'id' 不能使用。"
            )
        # Map True to empty array
        relationship_properties_mapped: List[str] = (
            [] if relationship_properties is True else relationship_properties
        )

        class RelationshipProperty(BaseModel):
            """A single property consisting of key and value"""

            key: str = optional_enum_field(
                relationship_properties_mapped,
                # description="Property key.",
                description="属性键。",
                input_type="property",
                llm_type=llm_type,
            )
            value: str = Field(..., description="属性值。") #  description="value")

        relationship_fields["properties"] = (
            Optional[List[RelationshipProperty]],
            # Field(None, description="List of relationship properties"),
            Field(None, description="关系属性列表。"),
        )
    SimpleRelationship = create_model("SimpleRelationship", **relationship_fields)  # type: ignore

    class DynamicGraph(_Graph):
        """Represents a graph document consisting of nodes and relationships."""

        # nodes: Optional[List[SimpleNode]] = Field(description=description="List of nodes")  # type: ignore
        nodes: Optional[List[SimpleNode]] = Field(description="节点列表")#description="List of nodes")  # type: ignore
        relationships: Optional[List[SimpleRelationship]] = Field(  # type: ignore
            # description="List of relationships"
            description="关系列表"
        )

    return DynamicGraph


def map_to_base_node(node: Any) -> Node:
    """Map the SimpleNode to the base Node."""
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)


def map_to_base_relationship(rel: Any) -> Relationship:
    """Map the SimpleRelationship to the base Relationship."""
    source = Node(id=rel.source_node_id, type=rel.source_node_type)
    target = Node(id=rel.target_node_id, type=rel.target_node_type)
    properties = {}
    if hasattr(rel, "properties") and rel.properties:
        for p in rel.properties:
            properties[format_property_key(p.key)] = p.value
    return Relationship(
        source=source, target=target, type=rel.type, properties=properties
    )


def _parse_and_clean_json(
    argument_json: Dict[str, Any],
) -> Tuple[List[Node], List[Relationship]]:
    nodes = []
    for node in argument_json["nodes"]:
        if not node.get("id"):  # Id is mandatory, skip this node
            continue
        node_properties = {}
        if "properties" in node and node["properties"]:
            for p in node["properties"]:
                node_properties[format_property_key(p["key"])] = p["value"]
        nodes.append(
            Node(
                id=node["id"],
                type=node.get("type"),
                properties=node_properties,
            )
        )
    relationships = []
    for rel in argument_json["relationships"]:
        # Mandatory props
        if (
            not rel.get("source_node_id")
            or not rel.get("target_node_id")
            or not rel.get("type")
        ):
            continue

        # Node type copying if needed from node list
        if not rel.get("source_node_type"):
            try:
                rel["source_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["source_node_id"]
                ][0]
            except IndexError:
                rel["source_node_type"] = None
        if not rel.get("target_node_type"):
            try:
                rel["target_node_type"] = [
                    el.get("type")
                    for el in argument_json["nodes"]
                    if el["id"] == rel["target_node_id"]
                ][0]
            except IndexError:
                rel["target_node_type"] = None

        rel_properties = {}
        if "properties" in rel and rel["properties"]:
            for p in rel["properties"]:
                rel_properties[format_property_key(p["key"])] = p["value"]

        source_node = Node(
            id=rel["source_node_id"],
            type=rel["source_node_type"],
        )
        target_node = Node(
            id=rel["target_node_id"],
            type=rel["target_node_type"],
        )
        relationships.append(
            Relationship(
                source=source_node,
                target=target_node,
                type=rel["type"],
                properties=rel_properties,
            )
        )
    return nodes, relationships


def _format_nodes(nodes: List[Node]) -> List[Node]:
    return [
        Node(
            id=el.id.title() if isinstance(el.id, str) else el.id,
            type=el.type.capitalize()  # type: ignore[arg-type]
            if el.type
            else None,  # handle empty strings  # type: ignore[arg-type]
            properties=el.properties,
        )
        for el in nodes
    ]


def _format_relationships(rels: List[Relationship]) -> List[Relationship]:
    return [
        Relationship(
            source=_format_nodes([el.source])[0],
            target=_format_nodes([el.target])[0],
            type=el.type.replace(" ", "_").upper(),
            properties=el.properties,
        )
        for el in rels
    ]


def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def _convert_to_graph_document(
    raw_schema: Dict[Any, Any],
) -> Tuple[List[Node], List[Relationship]]:
    # If there are validation errors
    if not raw_schema["parsed"]:
        try:
            try:  # OpenAI type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["tool_calls"][0]["function"][
                        "arguments"
                    ]
                )
            except Exception:  # Google type response
                argument_json = json.loads(
                    raw_schema["raw"].additional_kwargs["function_call"]["arguments"]
                )

            nodes, relationships = _parse_and_clean_json(argument_json)
        except Exception:  # If we can't parse JSON
            return ([], [])
    else:  # If there are no validation errors use parsed pydantic object
        parsed_schema: _Graph = raw_schema["parsed"]
        nodes = (
            [map_to_base_node(node) for node in parsed_schema.nodes if node.id]
            if parsed_schema.nodes
            else []
        )

        relationships = (
            [
                map_to_base_relationship(rel)
                for rel in parsed_schema.relationships
                if rel.type and rel.source_node_id and rel.target_node_id
            ]
            if parsed_schema.relationships
            else []
        )
    # Title / Capitalize
    return _format_nodes(nodes), _format_relationships(relationships)


class LLMGraphTransformer:
    """Transform documents into graph-based documents using a LLM.

    It allows specifying constraints on the types of nodes and relationships to include
    in the output graph. The class supports extracting properties for both nodes and
    relationships.

    Args:
        llm (BaseLanguageModel): An instance of a language model supporting structured
          output.
        allowed_nodes (List[str], optional): Specifies which node types are
          allowed in the graph. Defaults to an empty list, allowing all node types.
        allowed_relationships (List[str], optional): Specifies which relationship types
          are allowed in the graph. Defaults to an empty list, allowing all relationship
          types.
        prompt (Optional[ChatPromptTemplate], optional): The prompt to pass to
          the LLM with additional instructions.
          传递给大型语言模型（LLM）的提示（prompt），其中可以包含执行任务所需的额外指令。
            如果提供了 prompt，它将指导语言模型如何理解和处理输入的文本。
        strict_mode (bool, optional): Determines whether the transformer should apply
          filtering to strictly adhere to `allowed_nodes` and `allowed_relationships`.
          Defaults to True.
          将应用过滤，严格遵循 allowed_nodes 和 allowed_relationships 参数中定义的节点和关系类型。
           False，则转换器可能会提取不在这些允许列表中的节点和关系类型。
        node_properties (Union[bool, List[str]]): If True, the LLM can extract any
          node properties from text. Alternatively, a list of valid properties can
          be provided for the LLM to extract, restricting extraction to those specified.
          可以是布尔值 True： 则表示语言模型可以从文本中提取任何节点属性。
            或一个字符串列表：字符串列表，则表示语言模型在提取时仅限于列表中指定的属性。
        relationship_properties (Union[bool, List[str]]): If True, the LLM can extract
          any relationship properties from text. Alternatively, a list of valid
          properties can be provided for the LLM to extract, restricting extraction to
          those specified.
           True，则语言模型可以从文本中提取任何关系属性。
           如果提供一个字符串列表，则表示语言模型在提取时仅限于列表中指定的属性。

    Example:
        .. code-block:: python
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            from langchain_core.documents import Document
            from langchain_openai import ChatOpenAI

            llm=ChatOpenAI(temperature=0)
            transformer = LLMGraphTransformer(
                llm=llm,
                allowed_nodes=["Person", "Organization"])

            doc = Document(page_content="Elon Musk is suing OpenAI")
            graph_documents = transformer.convert_to_graph_documents([doc])
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        allowed_nodes: List[str] = [],
        allowed_relationships: List[str] = [],
        prompt: Optional[ChatPromptTemplate] = None,
        strict_mode: bool = True,
        node_properties: Union[bool, List[str]] = False,
        relationship_properties: Union[bool, List[str]] = False,
    ) -> None:
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.strict_mode = strict_mode
        self._function_call = True
        # Check if the LLM really supports structured output       # True 。最终的输出始终是一个包含键 "raw"、"parsed" 和 "parsing_error" 的字典。
        try:
            llm.with_structured_output(_Graph, include_raw=True)  #  这里导致了错误？\ include_raw False，则只返回解析后的结构化输出。如果在模型输出解析过程中发生错误，将会抛出异常
        except NotImplementedError:
            self._function_call = False
        if not self._function_call:
            if node_properties or relationship_properties:
                raise ValueError(
                    "The 'node_properties' and 'relationship_properties' parameters "
                    "cannot be used in combination with a LLM that doesn't support "
                    "native function calling."
                )
            try:
                import json_repair  # type: ignore

                self.json_repair = json_repair
            except ImportError:
                raise ImportError(
                    "Could not import json_repair python package. "
                    "Please install it with `pip install json-repair`."
                )
            prompt = prompt or create_unstructured_prompt(
                allowed_nodes, allowed_relationships
            )
            self.chain = prompt | llm
        else:
            # Define chain
            try:
                llm_type = llm._llm_type  # type: ignore
            except AttributeError:
                llm_type = None
            schema = create_simple_model(
                allowed_nodes,
                allowed_relationships,
                node_properties,
                llm_type,
                relationship_properties,
            ) # DynamicGraph
            structured_llm = llm.with_structured_output(schema, include_raw=True)
            prompt = prompt or default_prompt
            self.chain = prompt | structured_llm

    def process_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Processes a single document, transforming it into a graph document using
        an LLM based on the model's schema and constraints.
        """
        text = document.page_content  # maybe cause erroe
        raw_schema = self.chain.invoke({"input": text}, config=config) # raw_schema tokenusage {'completion_tokens': 1024, 'prompt_tokens': 2170, 'total_tokens': 3194}
        if self._function_call:
            if not isinstance(raw_schema, dict):
                raw_schema = {'parsed': raw_schema, 'raw': raw_schema}
            raw_schema = cast(Dict[Any, Any], raw_schema)      #  "raw"、"parsed" 和 "parsing_error" 的字典 raw_schema = {'parsed': raw_schema, 'raw': raw_schema}
            nodes, relationships = _convert_to_graph_document(raw_schema)
        else:
            nodes_set = set()
            relationships = []
            if not isinstance(raw_schema, str):
                raw_schema = raw_schema.content
            parsed_json = self.json_repair.loads(raw_schema)
            if not isinstance(parsed_json, list):
                parsed_json = [parsed_json]
            for rel in parsed_json:
                # Ensure that 'head_type' and 'tail_type' keys exist
                node_keys = ['head', 'head_type', 'tail', 'tail_type', 'relation']
                is_bad_relationship = False
                for node_key in node_keys:
                    if node_key not in rel:
                        # print(f">> Missing key '{node_key}' in relationship: {rel}")
                        is_bad_relationship = True

                # Nodes need to be deduplicated using a set
                # if 'head' in rel and 'head_type' in rel:
                # if 'tail' in rel and 'tail_type' in rel:
                
                if is_bad_relationship:
                    print(f'## warning the parsed_json is {parsed_json} \n ##--> bad relation is {rel}')
                    continue

                nodes_set.add((rel["head"], rel["head_type"]))
                nodes_set.add((rel["tail"], rel["tail_type"]))

                source_node = Node(id=rel["head"], type=rel["head_type"])
                target_node = Node(id=rel["tail"], type=rel["tail_type"])
                relationships.append(
                    Relationship(
                        source=source_node, target=target_node, type=rel["relation"]
                    )
                )
            # Create nodes list
            nodes = [Node(id=el[0], type=el[1]) for el in list(nodes_set)]
        nodes_bk = copy.deepcopy(nodes)
        relationships_bk = copy.deepcopy(relationships)

        # Strict mode filtering
        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        # TODO 增加校验~  跳过一些错误的信息
        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    def convert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """Convert a sequence of documents into graph documents.

        Args:
            documents (Sequence[Document]): The original documents.
            **kwargs: Additional keyword arguments.

        Returns:
            Sequence[GraphDocument]: The transformed documents as graphs.
        """
        return [self.process_response(document, config) for document in documents]

    async def aprocess_response(
        self, document: Document, config: Optional[RunnableConfig] = None
    ) -> GraphDocument:
        """
        Asynchronously processes a single document, transforming it into a
        graph document.
        """
        text = document.page_content
        raw_schema = await self.chain.ainvoke({"input": text}, config=config)
        raw_schema = cast(Dict[Any, Any], raw_schema)
        nodes, relationships = _convert_to_graph_document(raw_schema)

        if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
            if self.allowed_nodes:
                lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
                nodes = [
                    node for node in nodes if node.type.lower() in lower_allowed_nodes
                ]
                relationships = [
                    rel
                    for rel in relationships
                    if rel.source.type.lower() in lower_allowed_nodes
                    and rel.target.type.lower() in lower_allowed_nodes
                ]
            if self.allowed_relationships:
                relationships = [
                    rel
                    for rel in relationships
                    if rel.type.lower()
                    in [el.lower() for el in self.allowed_relationships]
                ]

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)

    async def aconvert_to_graph_documents(
        self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
    ) -> List[GraphDocument]:
        """
        Asynchronously convert a sequence of documents into graph documents.
        """
        tasks = [
            asyncio.create_task(self.aprocess_response(document, config))
            for document in documents
        ]
        results = await asyncio.gather(*tasks)
        return results
