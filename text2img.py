from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch


model_id = 'stabilityai/stable-diffusion-2-1'

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
# prompt = "a photo of an anstronaut riding a horse"
prompt = "A fox in watercolor painting style"
#prompt = "a phone of a beautiful chinese girl riding a horse"
n_prompt= "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, cartoon, ugly, deformed"
image = pipe(prompt, negative_prompt=n_prompt).images[0]

image.save("rides_horse.png")

