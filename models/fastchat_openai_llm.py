from abc import ABC
import requests
from typing import Optional, List
from langchain.llms.base import LLM

from models.loader import LoaderCheckPoint
from models.base import (RemoteRpcModel,
                         AnswerResult)
from typing import (
    Collection,
    Dict
)
import tiktoken


def get_token_len(test_str):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k-0613")
    result = enc.encode(test_str)
    return len(result)


def _build_message_template() -> Dict[str, str]:
    """
    :return: 结构
    """
    return {
        "role": "",
        "content": "",
    }


class FastChatOpenAILLM(RemoteRpcModel, LLM, ABC):
    api_base_url: str = "http://localhost:8000/v1"
    model_name: str = "chatglm-6b"
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "FastChat"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    @property
    def _api_key(self) -> str:
        pass

    @property
    def _api_base_url(self) -> str:
        return self.api_base_url

    def set_api_key(self, api_key: str):
        pass

    def set_api_base_url(self, api_base_url: str):
        self.api_base_url = api_base_url

    def call_model_name(self, model_name):
        self.model_name = model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    # 将历史对话数组转换为文本格式
    def build_message_list(self, query, history) -> Collection[Dict[str, str]]:
        if get_token_len(query) > 10000:  # 16k version
            query = '你好'
        build_message_list: Collection[Dict[str, str]] = []
        history = history[-self.history_len:] if self.history_len > 0 else []
        for i, (old_query, response) in enumerate(history):
            user_build_message = _build_message_template()
            user_build_message['role'] = 'user'
            user_build_message['content'] = old_query
            system_build_message = _build_message_template()
            system_build_message['role'] = 'assistant'
            system_build_message['content'] = response
            build_message_list.append(user_build_message)
            build_message_list.append(system_build_message)

        user_build_message = _build_message_template()
        user_build_message['role'] = 'user'
        user_build_message['content'] = query
        build_message_list.append(user_build_message)

        return build_message_list

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):

        try:
            import openai
            # export OPENAI_API_KEY="XXX" in .bashrc
            openai.api_base = self.api_base_url
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        if streaming:
            history += [[]]
            final_resp = ""
            for stream_resp in openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=self.build_message_list(prompt, history[:-1]),
                    stream=True,
                    max_tokens=2048
            ):
                final_resp += stream_resp.choices[0].delta.get('content', '')
                history[-1] = [prompt, final_resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": final_resp}
                yield answer_result
        else:
            # create a chat completion
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                messages=self.build_message_list(prompt, history),
                max_tokens=2048
            )

            history += [[prompt, completion.choices[0].message.content]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": completion.choices[0].message.content}

            yield answer_result
