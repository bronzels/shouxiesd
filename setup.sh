if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Mac detected."
    #mac
    os=darwin
    MYHOME=/Volumes/data
    SED=gsed
    bin=/Users/apple/bin
else
    echo "Assuming linux by default."
    #linux
    os=linux
    MYHOME=
    SED=sed
    bin=/usr/local/bin
fi

SD_HOME=${MYHOME}/workspace/shouxiellm/sd

cd ${SD_HOME}



######################start of stable-diffusion-webui-master######################
mkdir Stable-diffusion
cd Stable-diffusion
wget -c https://hf-mirror.com/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors?download=true -O v2-1_768-ema-pruned.safetensors

git clone git@github.com:AUTOMATIC1111/stable-diffusion-webui.git stable-diffusion-webui-master
cd stable-diffusion-webui-master

conda create -n sd-webui-master python=3.10 -y
conda activate sd-webui-master

#install git in higher version
cat >> ~/.bashrc <<EOF
export OMP_NUM_THREADS=\$(nproc --all)
export GOMP_CPU_AFFINITY=0-\$(( \$(nproc --all) - 1 ))
EOF
source ~/.bashrc
yum remove git -y
yum install curl-devel expat-devel gettext-devel openssl-devel zlib-devel -y
yum install perl-ExtUtils-MakeMaker -y
GIT_VERSION=2.45.0
wget -c https://github.com/git/git/archive/refs/tags/v2.45.0.tar.gz -O git-${GIT_VERSION}.tar.gz
tar -xzf git-${GIT_VERSION}.tar.gz
cd git-${GIT_VERSION}
#conda activate sd，要在虚拟环境下编译
make -j$(nproc --all) prefix=/usr/local all
make prefix=/usr/local install
cd ..
git version
yum install -y git-lfs wget tmux mesa-libGL gperftools-libs
echo "export LD_PRELOAD=/lib64/libtcmalloc.so.4" >> ~/.bashrc
source ~/.bashrc

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements_versions.txt
pip install xformers

cd stable-diffusion-webui-master
chmod a+x *.sh
git config --global --add safe.directory /workspace/shouxiellm/sd/stable-diffusion-webui-master
\cp ../webui-user.sh ./
:<<EOF
the following values were not passed to `accelerate launch` and had defaults used instead:
        `--num_processes` was set to a value of `1`
        `--num_machines` was set to a value of `1`
        `--mixed_precision` was set to a value of `'no'`
        `--dynamo_backend` was set to a value of `'no'`
EOF
file=webui.sh
cp ${file} ${file}.bk
$SED -i 's/accelerate launch --num_cpu_threads_per_process=6/accelerate launch --num_cpu_threads_per_process=6 --num_processes 12/g' ${file}
git config --global http.postBuffer 524288000
git config --global --add safe.directory /workspace/shouxiellm/sd/stable-diffusion-webui-master/repositories/stable-diffusion-webui-assets
git config --global --add safe.directory /workspace/shouxiellm/sd/stable-diffusion-webui-master/repositories/stable-diffusion-stability-ai
git config --global --add safe.directory /workspace/shouxiellm/sd/stable-diffusion-webui-master/repositories/generative-models
git config --global --add safe.directory /workspace/shouxiellm/sd/stable-diffusion-webui-master/repositories/k-diffusion
git config --global --add safe.directory /workspace/shouxiellm/sd/stable-diffusion-webui-master/repositories/BLIP
while ! ./webui.sh -f; do sleep 2 ; done ; echo succeed

rm -rf models/Stable-diffusion/
ln -s ${SD_HOME}/Stable-diffusion ${SD_HOME}/stable-diffusion-webui-master/models/Stable-diffusion

python3 launch.py -f --skip-torch-cuda-test --skip-version-check --listen --no-half --precision full

git clone https://github.com/VinsonLaro/stable-diffusion-webui-chinese
cp stable-diffusion-webui-chinese/localizations/chinese-all-0313.json  stable-diffusion-webui-master/localizations/chinese-all.json
cp stable-diffusion-webui-chinese/localizations/chinese-english-0313.json stable-diffusion-webui-master/localizations/chinese-english.json

pip install alibabacloud_imageseg20191230
export ALIYUN_ACCESS_KEY_ID=
export ALIYUN_ACCESS_KEY_SECRET=
######################end of stable-diffusion-webui-master######################

######################start of diffusers######################
conda create -n diffusers python=3.10 -y
conda activate diffusers

pip install -U huggingface_hub
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
source ~/.bashrc

pip install --upgrade diffusers[torch]
pip install transformers
######################end of diffusers######################

######################start of sd-ft######################
unset LD_PRELOAD
#tcmalloc会导致diffusers训练推理失败
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

conda create -n sd-ft python=3.10 -y
conda activate sd-ft

git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
cd examples
#export LD_PRELOAD=/lib64/libtcmalloc.so.4
yum install -y libaio-devel
export CFLAGS=$CFLAGS:/usr/include && export LDFLAGS=$LDFLAGS:/usr/lib64
pip install deepspeed diffusers[training]
accelerate config
:<<EOF
No distributed training
CPU only: NO
torch dynamo: NO
DeepSpeed: yes
  /workspace/shouxiellm/sd/ds_config_zero3.json
GPU: 0
FP16 or BF16: NO

从zero.json中删除optimizer因为
unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
result = self._prepare_deepspeed(*args)
ValueError: You cannot specify an optimizer in the config file and in the code at the same time. 
EOF
file=/workspace/hfcache/accelerate/default_config.yaml
cp ${file} ${file}.bk
#num_processes必须是1，不然GPU会从0增长
#install cutlass
git config --global credential.helper store
huggingface-cli login
######################end of sd-ft######################

######################start of sd-lora######################
pip install -r text_to_image/requirements.txt
#  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
#accelerate launch train_text_to_image_lora_sdxl.py \
#sdxl text_encoder_two.to GPU显存12G都占满不够了
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
#export DATASET_NAME="lambdalabs/pokemon-blip-captions"
#  --dataset_name=$DATASET_NAME --caption_column="text" \
#  --output_dir="sd-pokemon-model-lora" \
#  --report_to="wandb"
# wandb: Network error (TransientError), entering retry loop
# 需要代理不然validation生成的图片无法上传
accelerate launch --mixed_precision="fp16" text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="${SD_HOME}/watercolor" --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="${SD_HOME}/output_lora_watercolor"
#  --validation_prompt="A horse in watercolor painting style" \
#  --report_to="tensorboard"
# 每次validation都要重新加载模型太慢
tensorboard --logdir=${SD_HOME}/output_lora_wercolor --host=0.0.0.0

pip install -r text_to_image/requirements_sdxl.txt
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
#export DATASET_NAME="lambdalabs/pokemon-blip-captions"
#  --dataset_name=$DATASET_NAME --caption_column="text" \
#  --validation_prompt="cute dragon creature" --report_to="wandb" \
accelerate launch text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir="${SD_HOME}/watercolor" --caption_column="text" \
  --resolution=1024 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="${SD_HOME}/output_lora_sdxl_watercolor"
#deepspeed zero3
#[rank0]: torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU
######################end of sd-lora######################

######################start of sd-dreambooth######################
pip install -r dreambooth/requirements.txt
cd ${SD_HOME}
python download_dataset.py
cd -
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#export MODEL_NAME="SG161222/Realistic_Vision_V2.0"
#export INSTANCE_DIR="${SD_HOME}/dog-example"
export INSTANCE_DIR="${SD_HOME}/zacliu-example"
rm -rf ${INSTANCE_DIR}/.huggingface
export OUTPUT_DIR="${SD_HOME}/output_dreambooth_lora_zacliu"
#提示保存deepspeed checkpoint失败
#修改accelerate config文件，去掉FP16混合精度，不用deepspeed
#提示ValueError: Attempting to unscale FP16 gradients
#升级peft没用，修改accelerate config文件，去掉FP16混合精度
pip install --upgrade peft
#  --instance_prompt="a photo of sks dog" \
accelerate launch dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of zacliu boy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of zacliu boy in a bucket" \
  --validation_epochs=50 \
  --seed="0"
tensorboard --logdir=${OUTPUT_DIR} --host=0.0.0.0
######################end of sd-dreambooth######################

######################start of sd-controlnet######################
pip install -r controlnet/requirements.txt
pip install xformers bitsandbytes
export OUTPUT_DIR="${SD_HOME}/output_controlnet"
# --validation_prompt "High-quality close-up dslr photo of man wearing a hat with trees in the background" "Girl smiling, professional dslr photograph, dark background, studio lights, high quality" "Portrait of a clown face, oil on canvas, bittersweet expression" \
# --enable_xformers_memory_efficient_attention \
accelerate launch controlnet/train_controlnet.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=multimodalart/facesyntheticsspigacaptioned \
 --conditioning_image_column=spiga_seg \
 --image_column=image \
 --caption_column=image_caption \
 --resolution=512 \
 --learning_rate=1e-5 \
 --report_to="tensorboard" \
 --train_batch_size=4 \
 --num_train_epochs=3 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --set_grads_to_none
######################end of sd-controlnet######################

######################start of sd-textual_inversion######################
pip install -r textual_inversion/requirements.txt
pip install xformers
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="${SD_HOME}/cat_toy_example"
rm -rf ${DATA_DIR}/.huggingface
export OUTPUT_DIR="${SD_HOME}/output_textual_inversion_cat"
#  --enable_xformers_memory_efficient_attention \
#  --checkpointing_steps=5000 \
#  --validation_steps=5000 \
#  --gradient_accumulation_steps=4 \
#  --gradient_checkpointing \
accelerate launch textual_inversion/textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --report_to="tensorboard" \
  --output_dir="${OUTPUT_DIR}"
tensorboard --logdir=${OUTPUT_DIR} --port=6009 --host=0.0.0.0
######################end of sd-textual_inversion######################

######################end of ComfyUI######################
conda create -n comfyui python=3.10 -y
conda activate comfyui

while ! git clone https://github.com/comfyanonymous/ComfyUI.git; do sleep 2 ; done ; echo succeed
cd ComfyUI
pip install -r requirements.txt

######################end of ComfyUI######################
