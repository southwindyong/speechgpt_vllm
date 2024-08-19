import asyncio
import importlib
import inspect
import re
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Optional, Set

import fastapi
import uvicorn
from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm.envs as envs
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.cli_args import make_arg_parser
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest,
                                              DetokenizeRequest,
                                              DetokenizeResponse,
                                              EmbeddingRequest, ErrorResponse,
                                              TokenizeRequest,
                                              TokenizeResponse)
# yapf: enable
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from vllm.entrypoints.openai.serving_tokenization import (
    OpenAIServingTokenization)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION
#新添加

import torch
import torch.nn as nn
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
import soundfile as sf
from typing import List
import argparse
import logging
import json
from tqdm import tqdm
import sys
import os
import re
import traceback
from peft import PeftModel
sys.path.append('/remote-home/ycyuan/Speechgpt/SpeechGPT/speechgpt')
from utils.speech2unit.speech2unit import Speech2Unit
import transformers
from transformers import AutoConfig, LlamaForCausalLM, LlamaTokenizer, GenerationConfig
from vllm.entrypoints.openai.protocol import (CompletionLogProbs,
                                              CompletionRequest,
                                              CompletionResponse,
                                              CompletionResponseChoice,
                                              CompletionResponseStreamChoice,
                                              CompletionStreamResponse,
                                              UsageInfo)
device = torch.device('cuda')

engine: AsyncLLMEngine
engine_args: AsyncEngineArgs
openai_serving_chat: OpenAIServingChat
openai_serving_completion: OpenAIServingCompletion
openai_serving_embedding: OpenAIServingEmbedding
openai_serving_tokenization: OpenAIServingTokenization

TIMEOUT_KEEP_ALIVE = 5  # seconds

# engine: AsyncLLMEngine
# engine_args: AsyncEngineArgs
# openai_serving_chat: OpenAIServingChat
# openai_serving_completion: OpenAIServingCompletion
# openai_serving_embedding: OpenAIServingEmbedding
# openai_serving_tokenization: OpenAIServingTokenization

logger = init_logger('vllm.entrypoints.openai.api_server')

_running_tasks: Set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        task = asyncio.create_task(_force_log())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.remove)

    yield


router = APIRouter()


def mount_metrics(app: fastapi.FastAPI):
    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app())
    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile('^/metrics(?P<path>.*)$')
    app.routes.append(metrics_route)


@router.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@router.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    generator = await openai_serving_tokenization.create_tokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, TokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@router.post("/detokenize")
async def detokenize(request: DetokenizeRequest):
    generator = await openai_serving_tokenization.create_detokenize(request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        assert isinstance(generator, DetokenizeResponse)
        return JSONResponse(content=generator.model_dump())


@router.get("/v1/models")
async def show_available_models():
    models = await openai_serving_completion.show_available_models()
    return JSONResponse(content=models.model_dump())


@router.get("/version")
async def show_version():
    ver = {"version": VLLM_VERSION}
    return JSONResponse(content=ver)


@router.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        return JSONResponse(content=generator.model_dump())


class ModelManager:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir
        self.model = None

    def get_model(self):
        if self.model is None:
            self.model = Speech2Unit(ckpt_dir=self.ckpt_dir)
        return self.model
    

model_manager = ModelManager(ckpt_dir="/remote-home/ycyuan/Speechgpt/SpeechGPT/speechgpt/utils/speech2unit")

def preprocess(
    raw_text: str,
    s2u,
):
    processed_parts = []
    template= "[Human]: {question} <eoh>. [SpeechGPT]: "
    for part in raw_text.split("is input:"):
        if os.path.isfile(part.strip()) and os.path.splitext(part.strip())[-1] in [".wav", ".flac", ".mp4"]:
            processed_parts.append(s2u(part.strip(), merged=True))
        else:
            processed_parts.append(part)
    processed_text = "is input:".join(processed_parts)
    print("raw_text",raw_text)
    print("processed_text:",processed_text)
    meta_instruction="You are an AI assistant whose name is SpeechGPT.\n- SpeechGPT is a intrinsic cross-modal conversational language model that is developed by Fudan University.  SpeechGPT can understand and communicate fluently with human through speech or text chosen by the user.\n- It can perceive cross-modal inputs and generate cross-modal outputs.\n"
    prompt_seq =  template.format(question=processed_text)
    prompt_seq = meta_instruction + template.format(question=processed_text)
    return prompt_seq



def extract_text_between_tags(text, tag1='[SpeechGPT] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response


def posttext(
    response: str,
):

    question = extract_text_between_tags(response, tag1="[Human]", tag2="<eoh>")
    answer = extract_text_between_tags(response + '<eoa>', tag1=f"[SpeechGPT] :", tag2="<eoa>")
    tq = extract_text_between_tags(response, tag1="[SpeechGPT] :", tag2="; [ta]") if "[ta]" in response else ''
    ta = extract_text_between_tags(response, tag1="[ta]", tag2="; [ua]") if "[ta]" in response else ''
    ua = extract_text_between_tags(response + '<eoa>', tag1="[ua]", tag2="<eoa>") if "[ua]" in response else ''

    return {"question":question, "answer":answer, "textQuestion":tq, "textAnswer":ta, "unitAnswer":ua}

# def extract_speech_data(text):
#     # 匹配 <sosp> 到 <eosp> 之间的内容，如果没有 <eosp>，则匹配到文本末尾
#     pattern = re.compile(r'<sosp>(.*?)(?:<eosp>|(?=<sosp>))', re.DOTALL)
#     matches = pattern.findall(text)
#     return matches

def extract_speech_data(text):
    # 匹配 <sosp> 到 <eosp> 之间的内容，如果没有 <eosp>，则匹配到文本末尾
    # 使用非贪婪模式来匹配每一段数据
    pattern = re.compile(r'<sosp>(.*?)(?:<eosp>|(?=<sosp>))', re.DOTALL)
    matches = pattern.findall(text)
    
    # 处理没有 <eosp> 的情况，确保匹配到文本末尾的内容
    if '<eosp>' not in text:
        # 额外提取从最后一个 <sosp> 到文本末尾的内容
        last_sosp_index = text.rfind('<sosp>')
        if last_sosp_index != -1:
            tail_data = text[last_sosp_index + len('<sosp>'):]
            matches.append(tail_data)
    
    return matches

def postprocess(
    responses:str,
    output_dir
):
    vocoder_dir = "/remote-home/ycyuan/Speechgpt/SpeechGPT/speechgpt/utils/vocoder"
    vocoder_path = os.path.join(vocoder_dir, "vocoder.pt")
    vocoder_cfg = os.path.join(vocoder_dir, "config.json")
    with open(vocoder_cfg) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(vocoder_path, vocoder_cfg).to(device)
    # responses = [posttext(x) for x in responses]
    speechdata = extract_speech_data(responses)

    # print(responses)
    
    #save repsonses
    init_num = sum(1 for line in open(f"{output_dir}/responses.json", 'r')) if os.path.exists(f"{output_dir}/responses.json") else 0             
    # file_path = os.path.join(output_dir, 'responses.json')
    # if not os.path.exists(file_path):
    # # 如果文件夹不存在，可以创建它
    #     os.makedirs(output_dir, exist_ok=True)
    # # 创建一个空文件
    #     with open(file_path, 'w') as f:
    #         pass  # 只是创建文件，不写入内容
    # with open(f"{output_dir}/responses.json", 'a') as f:
    #     # if r["textAnswer"] != "":
    #     #     print("Transcript:", r["textQuestion"])
    #     #     print("Text response:", r["textAnswer"])
    #     # else:
    #     #     print("Response:\n", r["answer"])
    #     json_line = json.dumps(responses)
    #     f.write(json_line+'\n')

    #dump wav
    wav = torch.tensor(0)
    os.makedirs(f"{output_dir}/wav/", exist_ok=True)
    if not speechdata:  # 检查 speechdata 是否为空
        print("No speech data available.")
    else:
        for i, speech in enumerate(speechdata):
            if not speech:  # 检查 speech 是否为空
                print(f"Speech data at index {i} is empty, skipping.")
                continue
            # if response["answer"] != '' and '<sosp>' in response["answer"]:
            unit = [int(num) for num in re.findall(r'<(\d+)>', speech)]
            x = {
                    "code": torch.LongTensor(unit).view(1, -1).to(device),
                }
            wav = vocoder(x, True)
            sf.write(
                f"{output_dir}/wav/answer_{init_num+i}.wav",
                wav.detach().cpu().numpy(),
                16000,
            )
            print(f"Speech repsonse is saved in {output_dir}/wav/answer_{init_num+i}.wav")




# @router.post("/v1/completions/speechgpt")
@router.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    # 数据预处理
    output_dir = "/remote-home/ycyuan/Speechgpt/SpeechGPT/speechgpt/output"
    prompts = request.prompt
    # print("prompts:", prompts)
    s2u = model_manager.get_model()
    with torch.no_grad():
            #preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(preprocess(prompt,s2u))
    request.prompt = preprocessed_prompts


    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    print(generator)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        file_path = os.path.join(output_dir, 'responses.json')
        if not os.path.exists(file_path):
        # 如果文件夹不存在，可以创建它
            os.makedirs(output_dir, exist_ok=True)
        # 创建一个空文件
            with open(file_path, 'w') as f:
                pass  # 只是创建文件，不写入内容
        with open(f"{output_dir}/responses.json", 'a') as f:
            for choice in generator.choices:
                f.write(choice.text)
                print(f"Response json is saved in {output_dir}/responses.json")
                
        # 提取生成的文本
        generated_texts = [choice.text for choice in generator.choices]

        # 处理生成的文本
        for text in generated_texts:
            postprocess(text,output_dir) 

        return JSONResponse(content=generator.model_dump())

# @router.post("/v1/completions")
# async def create_completion(request: CompletionRequest, raw_request: Request):
#     generator = await openai_serving_completion.create_completion(
#         request, raw_request)
#     if isinstance(generator, ErrorResponse):
#         return JSONResponse(content=generator.model_dump(),
#                             status_code=generator.code)
#     if request.stream:
#         return StreamingResponse(content=generator,
#                                  media_type="text/event-stream")
#     else:
#         return JSONResponse(content=generator.model_dump())


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest, raw_request: Request):
    generator = await openai_serving_embedding.create_embedding(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    else:
        return JSONResponse(content=generator.model_dump())


def build_app(args):
    app = fastapi.FastAPI(lifespan=lifespan)
    app.include_router(router)
    app.root_path = args.root_path

    mount_metrics(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_, exc):
        err = openai_serving_chat.create_error_response(message=str(exc))
        return JSONResponse(err.model_dump(),
                            status_code=HTTPStatus.BAD_REQUEST)

    if token := envs.VLLM_API_KEY or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if request.method == "OPTIONS":
                return await call_next(request)
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    return app


def run_server(args, llm_engine=None):
    app = build_app(args)

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]

    global engine, engine_args

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = (llm_engine
              if llm_engine is not None else AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.OPENAI_API_SERVER))

    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    global openai_serving_chat
    global openai_serving_completion
    global openai_serving_embedding
    global openai_serving_tokenization

    openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names,
        args.response_role,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )
    openai_serving_completion = OpenAIServingCompletion(
        engine,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
    )
    openai_serving_embedding = OpenAIServingEmbedding(
        engine,
        model_config,
        served_model_names,
        request_logger=request_logger,
    )
    openai_serving_tokenization = OpenAIServingTokenization(
        engine,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        request_logger=request_logger,
        chat_template=args.chat_template,
    )
    app.root_path = args.root_path

    logger.info("Available routes are:")
    for route in app.routes:
        if not hasattr(route, 'methods'):
            continue
        methods = ', '.join(route.methods)
        logger.info("Route: %s, Methods: %s", route.path, methods)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/scripts.py for CLI entrypoints.
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    run_server(args)
