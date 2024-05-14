from diffusers import DiffusionPipeline
import torch

import time
start_time = time.time()

base_id = 'stabilityai/stable-diffusion-xl-base-1.0'
refiner_id = 'stabilityai/stable-diffusion-xl-refiner-1.0'

base = DiffusionPipeline.from_pretrained(base_id, 
                                         torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
base = base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(refiner_id,
                                            text_encoder=base.text_encoder_2,
                                            vae=base.vae,
                                            torch_dtype=torch.float16,
                                            use_safetensors=True,
                                            variant="fp16"
)
refiner = refiner.to("cuda")

n_stpes = 40
high_noise_frac = 0.8

#prompt = "A jajestic lion jumpying from a big stone at night"
prompt = "a photo of an anstronaut riding a horse"
#prompt = "A fox in watercolor painting style"
#prompt = "a phone of a beautiful chinese girl riding a horse"

infer_start_time = time.time()

image = base(
  prompt=prompt,
  num_inference_steps=n_stpes,
  denoising_strength=high_noise_frac,
  output_type="latent"
).images
image = refiner(
  prompt=prompt,
  num_inference_steps=n_stpes,
  denoising_strength=high_noise_frac,
  image=image
).images[0]


end_time = time.time()
print("infer耗时: {:.2f}秒".format(end_time - infer_start_time))
print("load+infer耗时: {:.2f}秒".format(end_time - start_time))

image.save("rides_horse.png")

'''
                                      load+infer time         infer time            gpu footprint
---------------------------------------------------------------------------------------------------------
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU
'''