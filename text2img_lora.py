from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = 'CompVis/stable-diffusion-v1-4'

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.unet.load_attn_procs("./output_lora_watercolor")
pipe = pipe.to("cuda")

prompt = "A fox in watercolor painting style"
#prompt = "a photo of an anstronaut riding a horse in watercolor painting style"
n_prompt= "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, cartoon, ugly, deformed"
image = pipe(prompt, negative_prompt=n_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

image.save("horse_watercolor.png")
#image.save("rides_horse_watercolor.png")

