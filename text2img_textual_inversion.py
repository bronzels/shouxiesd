from diffusers import StableDiffusionPipeline
import torch

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
repo_id_embeds = "./output_textual_inversion_cat"

pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.load_textual_inversion(repo_id_embeds)

prompt = "a <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("cat-toy-backpack.png")