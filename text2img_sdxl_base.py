from diffusers import DiffusionPipeline
import torch

import time
start_time = time.time()

model_id = 'stabilityai/stable-diffusion-xl-base-1.0'

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

#prompt = "a photo of an anstronaut riding a horse"
#prompt = "A fox in watercolor painting style"
#prompt = "a phone of a beautiful chinese girl riding a horse"
prompt = "Watercolor painting of a desert landscape, with sand dunes, mountains, and a blazing sun, soft and delicate brushstrokes, warm and vibrant colors"

infer_start_time = time.time()
image = pipe(prompt=prompt).images[0]

end_time = time.time()
print("infer耗时: {:.2f}秒".format(end_time - infer_start_time))
print("load+infer耗时: {:.2f}秒".format(end_time - start_time))

#image.save("rides_horse.png")
image.save("Watercolor_desert.png")
#pipe.save_pretrained("/workspace/shouxiellm/sd/Stable-diffusion/stabilityai/stable-diffusion-xl-base-1.0")
'''
                                      load+infer time         infer time            gpu footprint
---------------------------------------------------------------------------------------------------------
origin                                38.03s                  34.59s                7688M
torch.compile                         82.33s                  78.89s                7926M
pipe.enable_model_cpu_offload         39.58s                  37.01s                5924M
'''