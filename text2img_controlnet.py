from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "stabilityai/stable-diffusion-2-1-base"
controlnet_path = "./output_controlnet"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
  base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

control_image = load_image("./controlnet_condimg.png")
#prompt = "High-quality close-up dslr photo of man wearing a hat with trees in the background"
prompt = "Girl smiling, professional dslr photograph, dark background, studio lights, high quality"

generator = torch.manual_seed(0)
image = pipe(
  prompt, num_inference_steps=50, generator=generator, image=control_image
).images[0]
image.save("./uncanny_face.png")