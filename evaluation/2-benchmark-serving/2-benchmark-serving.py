"""Benchmark online serving throughput.
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Optional
import os
import sys

import aiohttp
import numpy as np
from tqdm import tqdm

from structs import TestRequest, Dataset, RequestResult
from backends import BACKEND_TO_PORTS
from distserve.tokenizer import get_tokenizer
from transformers import PreTrainedTokenizerBase

pbar: Optional[tqdm] = None

def smart_text_cut(text: str):
    ret = []

    length = len(text)
    cut_index = [int(length*0.8), int(length*0.7),
                 int(length*0.5), int(length*0.3), length]
    # cut_index = [int(length*0.8), int(length*0.7)]

    for index in cut_index:
        cut_index = text.rfind(' ', 0, index)
    
        if cut_index == -1:
            ret.append(text[:length])
        else:
            ret.append(text[:cut_index])
    
    return ret

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: str,
    fixed_output_len: Optional[int] = None,
) -> List[TestRequest]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Get tokenizer
    tokenizer = get_tokenizer(tokenizer,
                              trust_remote_code=args.trust_remote_code)
    
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[TestRequest] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests*5:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 16 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 128 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        # filtered_dataset.append(TestRequest(prompt, prompt_len, output_len))

        cut_prompts = smart_text_cut(prompt)
        for prompt in cut_prompts:
            prompt_token_ids = tokenizer(prompt).input_ids
            prompt_len = len(prompt_token_ids)
            filtered_dataset.append(TestRequest(prompt, prompt_len, output_len))
            
    return filtered_dataset

async def get_request(
    input_requests: List[TestRequest],
    process_name: str = "possion",
    request_rate: float = 1.0,
    cv: float = 1.0,
) -> AsyncGenerator[TestRequest, None]:
    interval_lens = len(input_requests)
    input_requests = iter(input_requests)

    if request_rate not in [float("inf"), 0.0]:
        if process_name == "uniform":
            intervals = [1.0 / request_rate for _ in range(interval_lens)]
        elif process_name == "gamma":
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        elif process_name == "possion":
            cv = 1
            shape = 1 / (cv * cv)
            scale = cv * cv / request_rate
            intervals = np.random.gamma(shape, scale, size=interval_lens)
        else:
            raise ValueError(
                f"Unsupported prosess name: {process_name}, we currently support uniform, gamma and possion."
            )
    for idx, request in enumerate(input_requests):
        yield request
        if request_rate == float("inf") or request_rate == 0.0:
            continue

        interval = intervals[idx]
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    verbose: bool
) -> RequestResult:
    global pbar
    if backend == "deepspeed":
        payload = {
            "prompt": prompt,
            "max_tokens": output_len,
            "min_new_tokens": output_len,
            "max_new_tokens": output_len,
            "stream": True,
            "max_length": int((prompt_len + output_len)*1.2+10) # *1.2 to prevent tokenization error
        }
        
        request_start_time = time.perf_counter()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3*3600)) as session:
            token_timestamps = []
            generated_text = ""
            try:
                async with session.post(url=api_url, json=payload) as response:
                    if response.status == 200:
                        async for data in response.content.iter_any():
                            token_timestamps.append(time.perf_counter())
                            try:
                                generated_text += json.loads(data.decode("utf-8")[6:])["text"][0]
                            except:
                                generated_text += data.decode("utf-8")
                        complete_time = time.perf_counter()
                    else:
                        print(response)
                        print(response.status)
                        print(response.reason)
                        sys.exit(1)
            except (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError) as e:
                print(e)
                sys.exit(1)
        request_end_time = time.perf_counter()
        
        if verbose:
            print(f"Prompt: {prompt}, Output: {generated_text}")
        
        pbar.update(1)
        return RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=token_timestamps,
            lifetime_events=None
        )
    else:
        headers = {"User-Agent": "Benchmark Client"}
        if backend == "distserve" or backend == "vllm":
            pload = {
                "prompt": prompt,
                "n": 1,
                "best_of": best_of,
                "use_beam_search": use_beam_search,
                "temperature": 0.0 if use_beam_search else 1.0,
                "top_p": 1.0,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": False,
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # The maximum length of the input is 2048, limited by the embedding
        # table size.
        assert prompt_len+output_len < 2048
        
        request_start_time = time.perf_counter()
        request_output = None

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                async with session.post(api_url, headers=headers, json=pload) as response:
                    chunks = []
                    async for chunk, _ in response.content.iter_chunks():
                        chunks.append(chunk)
                output = b"".join(chunks).decode("utf-8")
                try:
                    output = json.loads(output)
                except:
                    print("Failed to parse the response:")
                    print(output)
                    continue
                if verbose:
                    print(f"Prompt: {prompt}\n\nOutput: {output['text']}")

                # Re-send the request if it failed.
                if "error" not in output:
                    request_output = output
                    break
                else:
                    print(f"Failed to process the request: {output['error']}")
                    print(f"Resending the request: {pload}")

        request_end_time = time.perf_counter()
        
        pbar.update(1)        
        return RequestResult(
            prompt_len,
            output_len,
            request_start_time,
            request_end_time,
            token_timestamps=request_output["timestamps"],
            lifetime_events=request_output.get("lifetime_events", None)
        )


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[TestRequest],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    request_cv: float = 1.0,
    process_name: str = "possion",
    verbose: bool = False
) -> List[RequestResult]:
    tasks: List[asyncio.Task] = []
    async for request in get_request(
        input_requests, process_name, request_rate, request_cv
    ):
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                request.prompt,
                request.prompt_len,
                request.output_len,
                best_of,
                use_beam_search,
                verbose
            )
        )
        tasks.append(task)
    request_results = await asyncio.gather(*tasks)
    return request_results


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    input_requests = sample_requests(
        args.dataset, args.num_prompts, args.tokenizer
    )
    # input_prompt1 = "I'm attempting to independently measure the performance (e.g., latency, throughput, etc.) of the prefill and decode phases. Is there a way to achieve this? I have noticed a few benchmarks that measure end-to-end throughput and latency but do not provide separate metrics for each phase."
    # input_prompt2 = input_prompt1 + "I would greatly appreciate any guidance on profiling these two phases separately."
    # input_requests: List[TestRequest] = []
    # input_requests.append(TestRequest(input_prompt1, 
    #                                   len(input_prompt1), 200))
    # input_requests.append(TestRequest(input_prompt2, 
    #                                   len(input_prompt2), 200))
    print("Sampling done. Start benchmarking...")

    global pbar
    pbar = tqdm(total=args.num_prompts)
    benchmark_start_time = time.time()
    request_results = asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
            args.request_cv,
            args.process_name,
            args.verbose
        )
    )
    benchmark_end_time = time.time()
    pbar.close()
    
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput:")
    print(f"\t{args.num_prompts / benchmark_time:.2f} requests/s")
    print(f"\t{sum([req.prompt_len + req.output_len for req in input_requests]) / benchmark_time:.2f} tokens/s")
    print(f"\t{sum([req.output_len for req in input_requests]) / benchmark_time:.2f} output tokens/s")

    with open(args.output, "w") as f:
        json.dump(request_results, f, default=vars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument(
        "--backend", type=str, default="distserve", choices=["distserve", "vllm", "deepspeed"]
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the (preprocessed) dataset."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts-req-rates", type=str, required=True,
        help="[(num_prompts, request_rate), ...] where num_prompts is the number of prompts to process and request_rate is the number of requests per second.",
    )
    parser.add_argument(
        "--request-cv",
        type=float,
        default=1.0,
        help="the coefficient of variation of the gap between" "the requests.",
    )
    parser.add_argument(
        "--process-name",
        type=str,
        default="possion",
        choices=["possion", "gamma", "uniform"],
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    
    parser.add_argument(
        "--exp-result-root",
        type=str,
        default=None,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: env var EXP_RESULT_ROOT)"
    )
    parser.add_argument(
        "--exp-result-dir",
        type=str,
        required=True,
        help="Experiment result will be stored under folder <exp-result-root>/<exp-result-dir> (default: <model_name>-<dataset.name>)"
    )
    parser.add_argument(
        "--exp-result-prefix",
        type=str,
        default=None,
        help="Exp result file will be named as <exp-result-prefix>-<num-prompts>-<req-rate>.exp (default: <backend>)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="tokenizer is used to convert dataset"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose logs (prompts and outputs)."
    )
    
    args = parser.parse_args()
    
    if args.exp_result_root == None:
        if "EXP_RESULT_ROOT" not in os.environ:
            print(f"Error: EXP_RESULT_ROOT is not set in environment variables")
            sys.exit(1)
        args.exp_result_root = os.getenv("EXP_RESULT_ROOT")
        
    if args.exp_result_prefix == None:
        args.exp_result_prefix = args.backend
        
    if args.port == None:
        args.port = BACKEND_TO_PORTS[args.backend]
        
    num_prompts_request_rates = eval(args.num_prompts_req_rates)
    for (num_prompts, request_rate) in num_prompts_request_rates:
        print("===================================================================")
        print(f"Running with num_prompts={num_prompts}, request_rate={request_rate}")
        args.num_prompts = num_prompts
        args.request_rate = request_rate
        output_dir = os.path.join(args.exp_result_root, args.exp_result_dir)
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f"{args.exp_result_prefix}-{num_prompts}-{request_rate}.exp")
        main(args)
        time.sleep(1)
        