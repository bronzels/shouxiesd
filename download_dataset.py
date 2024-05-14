from huggingface_hub import snapshot_download

#local_dir = "./dog-example"
#dataset_id = "diffusers/dog-example"
local_dir = "./cat_toy_example"
dataset_id = "diffusers/cat_toy_example"
snapshot_download(
    dataset_id,
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
