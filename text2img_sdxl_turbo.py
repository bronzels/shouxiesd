from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

#prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
prompt = "a photo of an anstronaut riding a horse"
steps = 1
#steps = 20
#steps = 5
#steps = 3
#steps = 2
#1,2步生成的不错
#20步生成的图像像版画

image = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]
image.save("./sdxl_turbo_baby_racoon.png")