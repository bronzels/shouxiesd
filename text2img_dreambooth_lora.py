from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.load_lora_weights("./output_dreambooth_lora_zacliu")
pipe = pipe.to("cuda")
#prompt = "A photo of sks dog in a chinese temple"
#prompt = "A photo of sks dog in a bucket"
#prompt = "A photo of zacliu boy in a bucket"
#prompt = "A photo of zacliu boy in a bmw car"
prompt = "A photo of zacliu boy playing with his mother"
n_prompt= "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, cartoon, ugly, deformed"
image = pipe(prompt, negative_prompt=n_prompt).images[0]

image.save("zacliu_boy.png")

