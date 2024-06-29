import os
from dataclasses import asdict
import streamlit as st
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from interface import GenerationConfig, generate_interactive
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from word2pic import pic
from ImageUnderstanding import under
from PIL import Image

logger = logging.get_logger(__name__)

class InternLM_LLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    
    def __init__(self, model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.model = self.model.eval()

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        system_prompt =  """
            你是古籍解读助手，精通古代文献和典籍，可以提供关于古籍解读、古文翻译和历史文化背景的专业建议和信息。无论是想了解古籍中的智慧，还是寻找特定古文的翻译和注释，都能提供帮助。
            """

        messages = [(system_prompt, '')]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "InternLM"

def load_chain(model, tokenizer):
    embeddings = HuggingFaceEmbeddings(model_name="/group_share/model/sentence-transformer")
    persist_directory = '/group_share/cmed2/data/data_base/vector_db/chroma'
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    retriever_chroma = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = InternLM_LLM(model, tokenizer)
    template = """你可以参考以下上下文进行思考，并回答最后的问题。不要表明思考过程，直接返回答案。如果你不知道答案，就说你不知道，不要试图编造答
    案。请提供详细并且结构清晰的回答，并尽量避免简单带过问题。
    {context}
    问题: {question}
    有用的回答:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_chroma, return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain

def on_btn_click():
    del st.session_state.messages

@st.cache_resource
def load_model():
    model = (
        AutoModelForCausalLM.from_pretrained("/group_share/cmed2/config/work_dirs/hf_merge", trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained("/group_share/cmed2/config/work_dirs/hf_merge", trust_remote_code=True)
    return model, tokenizer

def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)
        enable_rag = st.checkbox('RAG检索')
        selected_page = st.radio("Select Page", ("文本生成", "图片生成", "语音识别", "图片理解", "文本解析"))

    generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature, repetition_penalty=1.002)
    return generation_config, enable_rag, selected_page



user_prompt = 'user\n{user}\n'
robot_prompt = 'assistant\n{robot}\n'
cur_query_prompt = 'user\n{user}\nassistant\n'

def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('You are InternLM (书生·浦语), a helpful, honest, '
                        'and harmless AI assistant developed by Shanghai '
                        'AI Laboratory (上海人工智能实验室).')
    total_prompt = f"<s>system\n{meta_instruction}\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt

def main():
    model, tokenizer = load_model()
    qa_chain = load_chain(model, tokenizer)
    st.title("甘肃政法大学古籍解读助手")
    
    generation_config, enable_rag, selected_page = prepare_generation_config()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("assistant")):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user", avatar='user',):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": 'user'})

        if enable_rag:
            with st.chat_message("robot", avatar='assistant',):
                message_placeholder = st.empty()
                cur_response = qa_chain({"query": prompt})["result"]
                message_placeholder.markdown(cur_response)
        else:
            with st.chat_message("robot", avatar='assistant'):
                message_placeholder = st.empty()
                for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
                ):
                    message_placeholder.markdown(cur_response + "▌")
                message_placeholder.markdown(cur_response)

        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,
            'avatar': 'assistant'
            
        })
        torch.cuda.empty_cache()

    if selected_page == "文本生成":
        page_1()
    elif selected_page == "图片生成":
        page_2()
    elif selected_page == "语音识别":
        page_3()
    elif selected_page == "图片理解":
        page_4()
    elif selected_page == "文本解析":
        page_5()

def page_1():
    # st.write("文本生成！！！")
    pass

def page_2():
    def add_to_history(message, image_path):
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"message": message, "image": image_path})

    def get_history():
        if "history" not in st.session_state:
            st.session_state.history = []
        return st.session_state.history
    message = st.chat_input('请输入要生成图片的文本:', key="page_2_chat_input")
    if message:
        save_path = pic(message)
        st.image(save_path, caption="生成的图片", use_column_width=True)
        add_to_history(message, save_path)
    else:
        st.write(" ")

def page_3():
    st.write("语音识别功能即将上线。")

def page_4(): 
    mess = st.chat_input('请输入问题:', key="page_4_chat_input")
    uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"], key="page_4_file_uploader")
    if mess and uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='上传的图片', use_column_width=True)
        pic_path = f'img/{uploaded_file.name}'
        image.save(pic_path)
        answer = under(mess, pic_path)
        st.write("图片理解的答案:", answer)
        os.remove(pic_path)

def page_5():
    # 获取用户输入的问题
    mess5 = st.chat_input('请输入问题:', key="page_5_chat_input")
    # 上传文本文件
    uploaded_file = st.file_uploader("上传文本文件", type=["txt"], key="page_5_file_uploader")
    if mess5 and uploaded_file:
        # 读取上传的文本文件内容
        file_content = uploaded_file.read().decode("utf-8")
        # 将文本内容和问题传递给模型进行解析
        response = parse_text(file_content, mess5)
        # 显示解析结果
        st.write("解析结果:", response)
    # 显示当前上传的文件内容
    if uploaded_file:
        file_content = uploaded_file.read().decode("utf-8")
        st.text_area("文件内容", file_content, height=300)

def parse_text(text, question):
    # 创建一个简单的Prompt模板
    prompt_template = """
    你是一个文本解析助手，负责从给定的文本中提取有用的信息。以下是提供的文本内容和需要回答的问题。请提供一个详细和准确的答案。
    文本内容:
    {text}
    问题:
    {question}
    回答:
    """
    # 格式化Prompt
    prompt = prompt_template.format(text=text, question=question)
    # 使用模型进行生成
    response = llm(prompt, generation_config)
    return response

def llm(prompt):
    # 加载模型和tokenizer
    model, tokenizer = load_model()
    # 调用模型进行预测
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=generation_config.max_length)
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    main()
