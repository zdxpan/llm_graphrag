from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Entities(BaseModel):
    """
    Identify and capture information about entities from text
    """

    names: List[str] = Field(
        description=
            "All the objects, person, organization, authors, keywords, title, abstract, institutions, citations, references, " + 
            "figures_tables, data, methods, results, discussion, conclusion, funding, classification_codes, proper_nouns, locations, " + 
            "time, laws_regulations, technical_terms, products_brands, species_taxonomy, codes_algorithms, statistical_indicators, " + 
            "research_topics, or business entities that appear in the text",
    )
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

# "authors, keywords, title, abstract, institutions, citations, references, figures_tables, data, methods, results, discussion, conclusion, funding, classification_codes, proper_nouns, locations, time, laws_regulations, technical_terms, products_brands, species_taxonomy, codes_algorithms, statistical_indicators, research_topics"
    # authors: List[str] = Field(
    #     description="The list of authors who contributed to the document."
    # )
    # keywords: List[str] = Field(
    #     description="Keywords that describe the main topics of the document."
    # )
    # title: str = Field(
    #     description="The title of the academic document."
    # )
    # abstract: str = Field(
    #     description="A brief summary of the document's content."
    # )
    # institutions: List[str] = Field(
    #     description="Institutions to which the authors are affiliated."
    # )
    # citations: List[str] = Field(
    #     description="References to other literature cited within the document."
    # )
    # references: List[str] = Field(
    #     description="A list of all the references at the end of the document."
    # )
    # figures_tables: List[str] = Field(
    #     description="Figures, tables, and images included in the document."
    # )
    # data: List[str] = Field(
    #     description="Numerical data or statistical information presented in the document."
    # )
    # methods: str = Field(
    #     description="The methods and techniques used in the research."
    # )
    # results: str = Field(
    #     description="The findings or results obtained from the research."
    # )
    # discussion: str = Field(
    #     description="Analysis and interpretation of the research results."
    # )
    # conclusion: str = Field(
    #     description="The final summary and key takeaways from the research."
    # )
    # funding: List[str] = Field(
    #     description="Sources of funding or grants that supported the research."
    # )
    # classification_codes: List[str] = Field(
    #     description="Classification codes such as library classification or subject field codes."
    # )
    # proper_nouns: List[str] = Field(
    #     description="Specific terms, theories, models, effects, etc., that are proper nouns in the field."
    # )
    # locations: List[str] = Field(
    #     description="Geographical locations relevant to the study or mentioned in case studies."
    # )
    # time: List[str] = Field(
    #     description="Time periods or historical eras relevant to the research."
    # )
    # laws_regulations: List[str] = Field(
    #     description="Laws, regulations, or standards cited in the document."
    # )
    # technical_terms: List[str] = Field(
    #     description="Professional jargon specific to a particular field."
    # )
    # products_brands: List[str] = Field(
    #     description="Products and brands mentioned in the research."
    # )
    # species_taxonomy: List[str] = Field(
    #     description="Names of species and taxonomic classifications used in biological research."
    # )
    # codes_algorithms: List[str] = Field(
    #     description="Code snippets and algorithm names in computer science research."
    # )
    # statistical_indicators: List[str] = Field(
    #     description="Statistical measures used for analysis, such as mean, standard deviation, etc."
    # )
    # research_topics: List[str] = Field(
    #     description="The main research topics or areas of study covered in the document."
    # )
