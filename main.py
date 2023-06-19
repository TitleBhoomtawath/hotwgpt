import openai
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, \
    StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import gradio as gr
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

max_input_size = 4096
num_outputs = 512
max_chunk_overlap = 20
model_name = 'gpt-3.5-turbo'
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name=model_name, max_tokens=num_outputs))

prompt_helper = PromptHelper(max_input_size, num_outputs)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
storage_context = StorageContext.from_defaults(persist_dir="index")


def construct_index(directory_path):
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    print("constructing index")
    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.set_index_id('vector_index')
    index.storage_context.persist(persist_dir="index")

    return index


def chatbot(input_text):
    # load index
    chatbot.index = load_index_from_storage(service_context=service_context, storage_context=storage_context,
                                            index_id='vector_index')

    # initialize engine
    chatbot.query_engine = chatbot.index.as_query_engine()

    # accept query
    response = chatbot.query_engine.query(input_text)
    return response.response


if __name__ == '__main__':
    index = construct_index("docs")
    iface = gr.Interface(fn=chatbot,
                         inputs="text",
                         outputs="textbox",
                         title="My AI Chatbot")
    iface.launch(share=True)
