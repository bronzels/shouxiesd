import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0")
pipe = pipe.to("cuda")

image_path = "./Watercolor_desert.png"

init_image = load_image(image_path).convert("RGB")

prompt = "Watercolor painting of a desert landscape, with sand dunes, mountains, and a blazing sun, soft and delicate brushstrokes, warm and vibrant colors"

negative_prompt = "(EasyNegative),(watermark), (signature), (sketch by bad-artist), (signature), (worst quality), (low quality), (bad anatomy), NSFW, nude, (normal quality)"

seed = torch.Generator("cuda").manual_seed(42)

image = pipe(prompt, negative_prompt=negative_prompt, generator=seed, image=init_image).images[0]

image.save("Watercolor_desert-refiner.png")