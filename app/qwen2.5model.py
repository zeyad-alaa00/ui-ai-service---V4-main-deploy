from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
    local_dir="./qwen-vl",
    local_dir_use_symlinks=False
)
