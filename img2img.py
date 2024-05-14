from diffusers import AutoPipelineForImage2Image
import torch
import requests
from diffusers.utils import load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
  "runwayml/stable-diffusion-v1-5",
  torch_dtype=torch.float16,
  use_safetensors=True,
).to("cuda")
prompt = "a portrait of a dog wearing a pearl earring"
#prompt = "a portrait of a cute Chinese girl in a blue and white stripes polo T-shirt and a green shorts"
#potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.
pipeline.safety_checker = lambda images, clip_input: (images, None)

url = "./The_Girl_with_a_Pearl_Earring.png"
#url = "zacliu-example/2023-07-11 132952_seg.png"

image = load_image(url).convert("RGB")
image.thumbnail((768, 768))

image = pipeline(prompt, image, num_inference_stpes=200, strength=0.75, guidance_scale=10.5).images[0]
image.save("./The_dog_with_a_Pearl_Earring.png")