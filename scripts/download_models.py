from huggingface_hub import snapshot_download


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of models to download")
    args = parser.parse_args()
    models = ["Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-0.5B"]
    for i, model in enumerate(models):
        if args.limit is not None and i >= args.limit:
            break
        print(f"Downloading {model}...")
        snapshot_download(
            model,
            local_dir=f"./models/{model}",
            resume_download=True,
            local_dir_use_symlinks=False
        )
    
    from modelscope.hub.snapshot_download import snapshot_download
    models = ["LLM-Research/Meta-Llama-3.1-8B"]
    # download from modelscope
    for i, model in enumerate(models):
        print(f"Downloading {model}...")
        snapshot_download(
            model,
            local_dir=f"./models/{model}",
            ignore_file_pattern=["original/**"],
        )
