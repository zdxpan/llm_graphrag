# langchain pymupdf huggingface-hub sentence-transformers
# from paperqa import Settings, ask

print('在当前目录放置需要解答的文件pdf')
# (llm) ollama list
# NAME                    ID              SIZE    MODIFIED     
# llava:7b                8dd30f6b0cb1    4.7 GB  43 hours ago
# majx13/test:latest      bc0789d54213    4.4 GB  6 days ago  
# qwen2:latest            e0d4e1163c58    4.4 GB  6 days ago  
# yefx/minicpm3_4b:latest b9591b1f0e94    2.5 GB  7 days ago  
local_llm_config = {
    'model_list': [
        {
            'model_name': 'ollama/qwen2',
            'litellm_params': {
                'model': 'ollama/qwen2',
                'api_base': 'http://localhost:11434'
            }
        }
    ]
}

# this is the initail demo  use not succ yet
# answer = ask(
#     '使用中文回答，作者有哪些，谁提出了改进算法',
#     settings=Settings(
#         llm='ollama/qwen2',
#         llm_config = local_llm_config, 
#         summary_llm = 'ollama/qwen2',
#         summary_llm_config = local_llm_config,
#     )
# )


from paperqa import Docs
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from openai import AsyncOpenAI
from paperqa import Docs, LlamaEmbeddingModel, OpenAILLMModel


# have tried a few models
model = "llama3"
model = "qwen2"
llm = ChatOllama(model=model, base_url="http://localhost:11434")
local_client = AsyncOpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
    # api_key = "sk-no-key-required"
)
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model='bge-m3') # model=model)
embeddings.embed_query('nihaw')

# Demonstrate Ollama and langchain are working
print(llm.invoke("Who was the first US President?"))
print(llm.invoke("11.9 11.11 哪个大?"))


docs = Docs(client=local_client,
            # embedding_client=LlamaEmbeddingModel(name='llama'), #  object has no attribute 'embeddings'
            embedding="bge-m3:latest",  #  'model "nomic-embed-text" not found, try pulling it first'
            # embedding="langchain",
            # embedding_client=embeddings,  #  object has no attribute 'embeddings'
            llm_model=OpenAILLMModel(config=dict(model="qwen2", temperature=0.1, frequency_penalty=1.5, max_tokens=4096)),
            summary_llm_model=OpenAILLMModel(config=dict(model="qwen2", temperature=0.1, frequency_penalty=1.5, max_tokens=512,))
        )

# docs.add("多模态人机交互综述_陶建华.pdf")
docs.add("/home/dell/llm/graphrag_lang_neo4j/uploads/多模态人机交互综述_陶建华.pdf")
answer = docs.query("非视觉感知 跟多模态交互有啥关系")
print(answer)


