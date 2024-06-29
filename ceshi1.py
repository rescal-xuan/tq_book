import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
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



@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]


        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break



def load_chain(model,tokenizer):
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/group_share/model/sentence-transformer")
    # 向量数据库持久化路径
    persist_directory = '/group_share/cmed2/data/data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )
    retriever_chroma=vectordb.as_retriever(search_kwargs={"k": 3})
    # ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever_chroma], weights=[0.4, 0.6])
    # 加载自定义 LLM
    llm = InternLM_LLM(model,tokenizer)
    # 定义一个 Prompt Template
    template = """你可以参考以下上下文进行思考，并回答最后的问题。不要表明思考过程，直接返回答案。如果你不知道答案，就说你不知道，不要试图编造答
    案。请提供详细并且结构清晰的回答，并尽量避免简单带过问题。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_chroma, return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    return qa_chain



def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    model = (AutoModelForCausalLM.from_pretrained('/group_share/cmed2/config/work_dirs/hf_merge',
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained('/group_share/cmed2/config/work_dirs/hf_merge',
                                              trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=32768)
        top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.7, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)
        enable_rag=st.checkbox('RAG检索')
        enable_word2pic=st.checkbox('图片生成')

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config,enable_rag,enable_word2pic


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('You are InternLM (书生·浦语), a helpful, honest, '
                        'and harmless AI assistant developed by Shanghai '
                        'AI Laboratory (上海人工智能实验室).')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
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
    
    print('load model begin.')
    model, tokenizer = load_model()
    qa_chain = load_chain(model, tokenizer)
    print('load model end.')
    user_avator = '/group_share/cmed2/assets/user.png'
    robot_avator = '/group_share/cmed2/assets/robot.png'
    st.title('甘肃政法大学古籍解读助手')
    
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




if __name__ == '__main__':
    main()
