from dataclasses import dataclass
import os

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import gc
from vllm.sampling_params import SamplingParams
from vllm import LLM
from cs336_alignment.math_baseline import (
    evaluate_vllm,
    get_evaluation_sample_params,
    get_evaluation_samples,
)
from cs336_alignment.vllm_util import init_vllm, load_policy_into_vllm_instance
from tqdm import tqdm, trange
import numpy as np
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import torch.distributed as dist
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel  # type: ignore
import torch
import random
from cs336_alignment.sft_helper import tokenize_to_tensor
from cs336_alignment.sft import train_sft
from cs336_alignment.config import SftConfig, ExpertIterConfig, load_config_from_file
from cs336_alignment.logger import setup_logging, print_and_log
from cs336_alignment.extract import preprocess_text


def cleanup():
    logging.shutdown()
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


@dataclass
class QuestionMetaInfo:
    question_id: int = -1
    correct_count: int = 0
    sample_count: int = 0

    def accuracy(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return min(self.correct_count / (self.sample_count + 2), 0.99)


def rollout(
    sampling_params: SamplingParams,
    vllm: LLM,
    sft_prompts: list[str],
    sft_responses: list[str],
    sub_prompts: list[str],
    sub_ground_truths: list[str],
    sub_question_ids: list[int],
    question_meta_infos: dict[int, QuestionMetaInfo],
    use_all_positive: bool = False,
) -> int:
    outputs = vllm.generate(sub_prompts, sampling_params=sampling_params)
    correct_question_count = 0
    for output, ground_truth, prompt, question_id in tqdm(
        zip(outputs, sub_ground_truths, sub_prompts, sub_question_ids),
        total=len(sub_prompts),
        desc="Processing generated outputs",
        leave=False,
    ):
        assert len(output.outputs) == sample_batch_size
        if question_id not in question_meta_infos:
            meta_info = QuestionMetaInfo(question_id)
            question_meta_infos[question_id] = meta_info
        else:
            meta_info = question_meta_infos[question_id]

        min_len_resp: str = ""
        min_len = float("inf")
        correct_responses = []
        for resp in output.outputs:
            resp_text = resp.text
            reward_dict = r1_zero_reward_fn(resp_text, ground_truth)
            if reward_dict["reward"] <= 0:
                continue
            correct_responses.append(resp_text)
            resp_len = len(resp_text)
            if resp_len < min_len:
                min_len = resp_len
                min_len_resp = resp_text

        if len(correct_responses) > 0:
            correct_question_count += 1

        meta_info.sample_count = meta_info.sample_count // 2 + sample_batch_size
        meta_info.correct_count = meta_info.correct_count // 2 + len(correct_responses)

        if use_all_positive:
            if len(correct_responses) / sample_batch_size >= 0.9:
                continue
            for resp in correct_responses[:4]:  # limit to 4 samples per question
                sft_prompts.append(prompt)
                sft_responses.append(resp)
        elif min_len_resp != "":
            resp = min_len_resp
            sft_prompts.append(prompt)
            sft_responses.append(resp)

    if correct_question_count == 0:
        from cs336_alignment.extract import extract_ans

        print_and_log("Warning: No positive samples collected in this batch.")
        for i in range(4):
            prompt = sub_prompts[i]
            ground_truth = sub_ground_truths[i]
            sft_prompts.append(prompt)
            ans = extract_ans(ground_truth, False)
            sft_responses.append(f"{ground_truth} </think> <answer> {ans} </answer>")
    return correct_question_count


def expert_iter(
    expert_iter_config: ExpertIterConfig,
    vllm_batch_size: int,
    sampling_params: SamplingParams,
    vllm: LLM,
    question_meta_infos: dict[int, QuestionMetaInfo],
    sample_question_ids: list[int],
    sample_prompts: list[str],
    sample_ground_truths: list[str],
) -> tuple[list[str], list[str]]:
    sft_prompts = []
    sft_responses = []
    from cs336_alignment.extract import extract_ans
    if not expert_iter_config.do_rollout:
        for prompt, ground_truth, question_id in zip(sample_prompts, sample_ground_truths, sample_question_ids):
            sft_prompts.append(prompt)
            ans = extract_ans(ground_truth, False)
            sft_responses.append(f"{ground_truth} </think> <answer> {ans} </answer>")
            if question_id not in question_meta_infos:
                meta_info = QuestionMetaInfo(question_id)
                question_meta_infos[question_id] = meta_info
            else:
                meta_info = question_meta_infos[question_id]
            meta_info.sample_count += 1
            meta_info.correct_count += 1
        return sft_prompts, sft_responses

    correct_count = 0
    question_batch_size = len(sample_question_ids)
    for i in trange(0, question_batch_size, vllm_batch_size, desc="Generating batches"):
        sub_question_ids = sample_question_ids[i : i + vllm_batch_size]
        sub_prompts = sample_prompts[i : i + vllm_batch_size]
        sub_ground_truths = sample_ground_truths[i : i + vllm_batch_size]
        correct_count += rollout(
            sampling_params,
            vllm,
            sft_prompts,
            sft_responses,
            sub_prompts,
            sub_ground_truths,
            sub_question_ids,
            question_meta_infos,
            use_all_positive=expert_iter_config.use_all_positive,
        )
    assert len(sft_prompts) > 0, "No positive samples collected in this EI step."
    accuracy = correct_count / question_batch_size
    correct_question_total = sum(1 for meta in question_meta_infos.values() if meta.correct_count > 0)
    print_and_log(f"correct_count={correct_count} accuracy={accuracy:.2f} new_samples={len(sft_prompts)} correct_questions={correct_question_total}")
    return sft_prompts, sft_responses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-c", "--config", type=str, required=False, help="Path to config file"
    )
    parser.add_argument("-t", "--test", action="store_true", help="Run in test mode")
    parser.add_argument(
        "--resume_from", type=int, default=0, help="EI step to resume from"
    )
    args = parser.parse_args()
    config = load_config_from_file(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    setup_logging(log_file_name=f"{config_name}.log")
    print_and_log("Starting SFT training...")
    print_and_log(f"Arguments: {args}")
    sft_config = SftConfig(**config.get("SftConfig", {}))
    expert_iter_config = ExpertIterConfig(**config.get("ExpertIterConfig", {}))
    print_and_log(f"SFT Config: {sft_config}")
    print_and_log(f"Expert Iter Config: {expert_iter_config}")

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_id = sft_config.model_id
    question_batch_size = expert_iter_config.question_batch_size
    vllm_batch_size = 64

    gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    if gpu_memory >= 64:
        vllm_batch_size = int(64 / 8 * vllm_batch_size)
    elif gpu_memory >= 32:
        vllm_batch_size = int(32 / 8 * vllm_batch_size)
    elif gpu_memory >= 16:
        vllm_batch_size = int(16 / 8 * vllm_batch_size)
    vllm_batch_size = min(vllm_batch_size, question_batch_size)
    print_and_log(f"Using vLLM batch size: {vllm_batch_size}")

    sample_batch_size = expert_iter_config.sample_batch_size
    n_ei_steps = expert_iter_config.n_ei_steps
    base_name = os.path.splitext(os.path.basename(args.config))[0]
    output_model = base_name

    prompts, ground_truths = get_evaluation_samples(sft_config.max_examples, 0)
    sampling_params = get_evaluation_sample_params(sample_batch_size, 2048 - 512 - 256)
    tokenizer = AutoTokenizer.from_pretrained(f"models/{model_id}")

    eval_sampling_params: SamplingParams = get_evaluation_sample_params(1, 2048 - 512 - 256)
    eval_prompts, eval_ground_truths = get_evaluation_samples(256, 4096)

    gpus_count = torch.cuda.device_count()
    vllm_device = "cuda:0" if gpus_count == 1 else "cuda:1"
    vllm = None
    train_device = "cuda:0"
    llm: PreTrainedModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    question_ids = np.array(range(len(prompts)))
    is_sample_device = vllm_device == train_device
    output_dir = f"models/{output_model}"
    os.makedirs(output_dir, exist_ok=True)
    if args.resume_from > 0:
        n_ei_steps = n_ei_steps - args.resume_from
        print_and_log(
            f"Resuming from EI step {args.resume_from}, remaining steps: {n_ei_steps}"
        )
        llm = AutoModelForCausalLM.from_pretrained(
            f"models/{output_model}",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": train_device},
        )
        llm.train()  # type: ignore

    question_meta_infos: dict[int, QuestionMetaInfo] = {}
    for ei_step in trange(args.resume_from, n_ei_steps, desc="Expert Iteration Steps"):
        correct_question_total = 0
        for idx in question_ids:
            if idx in question_meta_infos:
                meta_info = question_meta_infos[idx]
                if meta_info.correct_count > 0:
                    correct_question_total += 1
        print_and_log(
            f"EI Step {ei_step + 1}/{n_ei_steps}: {correct_question_total}/{len(question_ids)} questions have correct samples."
        )
        # Sample indices for this EI step
        sample_question_ids = np.random.choice(question_ids, size=question_batch_size, replace=False)
        sample_question_ids = list(sample_question_ids)
        assert len(sample_question_ids) == question_batch_size

        sample_prompts = [prompts[j] for j in sample_question_ids]
        sample_ground_truths = [ground_truths[j] for j in sample_question_ids]
        assert len(sample_prompts) == question_batch_size
        assert len(sample_ground_truths) == question_batch_size

        if vllm is None:
            print("Initializing vLLM model for EI step...")
            vllm = init_vllm(
                model_id=f"models/{model_id}",
                device=vllm_device,
                seed=42,
                gpu_memory_utilization=0.85,
            )
        if llm is not None:
            assert vllm is not None, "vLLM should be initialized"
            print_and_log("Loading policy weights into vLLM model...")
            load_policy_into_vllm_instance(llm, vllm)

            print_and_log("evaluating vLLM on sample sets...")
            evaluate_vllm(vllm, sample_prompts, sample_ground_truths, eval_sampling_params)

            print_and_log("evaluating vLLM on eval sets...")
            evaluate_vllm(vllm, eval_prompts, eval_ground_truths, eval_sampling_params)

        sft_prompts, sft_responses = expert_iter(
            expert_iter_config=expert_iter_config,
            vllm_batch_size=vllm_batch_size,
            sampling_params=sampling_params,
            vllm=vllm,
            question_meta_infos=question_meta_infos,
            sample_question_ids=sample_question_ids,
            sample_prompts=sample_prompts,
            sample_ground_truths=sample_ground_truths,
        )

        # Free up vLLM memory if on the same device
        if is_sample_device:
            del vllm
            vllm = None
            print_and_log("Clearing vLLM from memory...")
            gc.collect()

        tokenized_data = tokenize_to_tensor(sft_prompts, sft_responses, tokenizer)
        input_ids = tokenized_data["input_ids"].to(train_device)
        response_mask = tokenized_data["response_mask"].to(train_device)
        labels = tokenized_data["labels"].to(train_device)
        print_and_log("Tokenization complete.")
        print_and_log(f"Input IDs shape: {input_ids.shape}")

        if args.test:
            break

        if llm is None:
            print_and_log("Loading model for SFT training...")
            llm = AutoModelForCausalLM.from_pretrained(
                f"models/{sft_config.model_id}",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": train_device},
            )
            llm.train()  # type: ignore

        if expert_iter_config.global_optimization:
            assert llm is not None
            if optimizer is None:
                optimizer = torch.optim.AdamW(
                    llm.parameters(), lr=sft_config.learning_rate, fused=True,
                    betas=(0.9, 0.95)
                )
        else:
            optimizer = None

        train_sft(
            sft_config,
            train_device,
            model=llm,  # type: ignore
            input_ids=input_ids,
            labels=labels,
            resp_mask=response_mask,
            print_entropy=True,
            optimizer=optimizer,
            use_lr_scheduler=not expert_iter_config.global_optimization,
        )

        if llm is not None and (ei_step + 1) % 2 == 0:
            print_and_log(
                f"{ei_step + 1}/{n_ei_steps} Saving fine-tuned model to {output_dir}..."
            )
            llm.save_pretrained(save_directory=output_dir)  # type: ignore
            if ei_step == 0:
                tokenizer.save_pretrained(save_directory=output_dir)

    llm.save_pretrained(save_directory=output_dir)  # type: ignore
    if vllm is None:
        vllm = init_vllm(
            model_id=output_dir,
            device=vllm_device,
            seed=42,
            gpu_memory_utilization=0.85,
        )
    evaluate_vllm(vllm, eval_prompts, eval_ground_truths, eval_sampling_params)
    cleanup()
