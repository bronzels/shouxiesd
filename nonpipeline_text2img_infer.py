from tqdm.auto import tqdm
from PIL import Image  
import torch  
from transformers import CLIPTextModel, CLIPTokenizer  
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler  
  
# 加载模型  
model_path = "runwayml/stable-diffusion-v1-5"  
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")  
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")  
text_encoder = CLIPTextModel.from_pretrained(  
 model_path, subfolder="text_encoder"  
)  
unet = UNet2DConditionModel.from_pretrained(  
 model_path, subfolder="unet"  
)  
scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
# 使用gpu加速  
torch_device = "cuda"  
vae.to(torch_device)  
text_encoder.to(torch_device)  
unet.to(torch_device)

# 对文本进行编码  
prompt = ["a photograph of an astronaut riding a horse on an unknown planet"]  
height = 512 # default height of Stable Diffusion  
width = 512 # default width of Stable Diffusion  
num_inference_steps = 25 # Number of denoising steps  
guidance_scale = 7.5 # Scale for classifier-free guidance  
batch_size = len(prompt)  
text_input = tokenizer(  
 prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"  
)  
with torch.no_grad():  
 text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

# 获取latent  
latents = torch.randn(  
 (batch_size, unet.config.in_channels, height // 8, width // 8),  
 device=torch_device,  
)  
latents = latents * scheduler.init_noise_sigma

# 降噪  
scheduler.set_timesteps(num_inference_steps)  
for t in tqdm(scheduler.timesteps):  
 latent_model_input = latents  
 latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)  
 with torch.no_grad():  
  # 预测噪声
  noise_pred = unet(
   latent_model_input, 
   t, 
   encoder_hidden_states=text_embeddings
  ).sample 
 # 降噪 
 latents = scheduler.step(noise_pred, t, latents).prev_sample

# 使用vae解码  
latents = 1 / 0.18215 * latents  
with torch.no_grad():  
 image = vae.decode(latents).sample  
 image = (image / 2 + 0.5).clamp(0, 1).squeeze()  
 image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()  
 images = (image * 255).round().astype("uint8")  
 image = Image.fromarray(image)  
image.save("./nonpipeline_rades_horse.png")

#相比pipeline画质糟糕很多


