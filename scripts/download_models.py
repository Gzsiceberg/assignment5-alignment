from huggingface_hub import snapshot_download


if __name__ == "__main__":
    models = ["Qwen/Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen1.5-0.5B"]
    for model in models:
        print(f"Downloading {model}...")
        snapshot_download(
            model,
            local_dir=f"./models/{model}",
            resume_download=True,
            local_dir_use_symlinks=False
        )
