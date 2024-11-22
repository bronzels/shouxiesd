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

SD_HOME=${MYHOME}/workspace/shouxiesd

cd ${SD_HOME}

MODEL_REPO=${SD_HOME}/Stable-diffusion


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
#增加set -e，循环安装避免git clone出错
$SED -i 's/accelerate launch --num_cpu_threads_per_process=6/accelerate launch --num_cpu_threads_per_process=6 --num_processes 12/g' ${file}
git config --global http.postBuffer 524288000
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/repositories/stable-diffusion-webui-assets
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/repositories/stable-diffusion-stability-ai
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/repositories/generative-models
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/repositories/k-diffusion
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/repositories/BLIP
while ! ./webui.sh -f; do sleep 2 ; done ; echo succeed

rm -rf models/Stable-diffusion/
ln -s ${SD_HOME}/Stable-diffusion ${SD_HOME}/stable-diffusion-webui-master/models/Stable-diffusion

HF_ENDPOINT=https://hf-mirror.com python3 launch.py -f --skip-torch-cuda-test --skip-version-check --listen --enable-insecure-extension-access --xformers
#--no-half --precision full

git clone https://github.com/VinsonLaro/stable-diffusion-webui-chinese
cp stable-diffusion-webui-chinese/localizations/chinese-all-0313.json  stable-diffusion-webui-master/localizations/chinese-all.json
cp stable-diffusion-webui-chinese/localizations/chinese-english-0313.json stable-diffusion-webui-master/localizations/chinese-english.json

pip install alibabacloud_imageseg20191230
export ALIYUN_ACCESS_KEY_ID=
export ALIYUN_ACCESS_KEY_SECRET=


#huggingface-cli download --resume-download Yntec/AbyssOrangeMix --local-dir ${MODEL_REPO}/Yntec_AbyssOrangeMix

:<<EOF
#prompt:
A girl, walking in the forest, th e sun fell on her body,
white dress,blonde hair,long hair,smiling,streching arms,hands up,beautiful,happy,
trees,bush,white flower,path,outdoor,
day,sunlight,blue sky,cloudy sky,
(masterpiece:1,2),best quality,ultra detailed,masterpiece,highres,8k,original,extremely detailed wallpaper,extremely detailed CG unity 8k wallpaper,perfect lighting,(extremely detailed CG:1.2),drawing,
painting,illustration,
anime,comic,game cg,
photorealistic,realistic,photography,
looking at viewer,facing the camera,close-up,upper body
#negative prompt:
NSFW,(worst quality:2),(low quality:2),(normal quality:2),lowres,normal quality,((monochrome)),((grayscale)),skin spots,acnes,skin blemishes,age spot,(ugly:1.331),(duplicate:1.331),(morbid:1.21),(mutilated:1.21),(tranny:1.331),mutated hands,(poorly drawn hands:1.5),blurry,(bad anatomy:1.21),(bad proportions:1.331),extra limbs,(disfigured:1.331),(missing arms:1.331),(extra legs:1.331),(fused fingers:1.61051),(too many fingers:1.61051),(unclear eyes:1.331),lowers,bad hands,missing fingers,extra digit,bad hands,missing fingers,(((extra arms and legs))),

提示词paitbrush画出了第三只手拿着画笔

step:20不管什么模型，出图都很模糊，step:50质量才能接受

#plugin
#download models from civitai C站模型下载
https://github.com/tzwm/sd-webui-model-downloader-cn
#prompt generation提示词反推
https://github.com/toriato/stable-diffusion-webui-wd14-tagger
#Images Browser：图库浏览器
https://github.com/AlUlkesh/stable-diffusion-webui-images-browser
#Tagcomplete：提示词自动补全/翻译
https://github.com/DominikDoom/a1111-sd-webui-tagcomplete
#model recommendation
#二次元
illustration,painting,sketch,drawing,paiting,comic,anime,catoon
Anything V5(动漫插画角色立绘)
Counterfeit(室内外场景，精致感溢出屏幕)
Dreamlike Diffusion（漫画插画风，梦幻，幻想，超现实魔幻主题作品）
Others:AbyssOrangeMix深渊橘,特立独行的DreamShaper,笔触细腻的Meina Mix和Cetus Mix, 魔幻风味Pastel Mix, 复古油画质感DalcefoPaiting
#真实系
photography,photo,realistic,photorealistic,RAW photo
Deliberate（超级升级版SD官方，非常真实质感，自由度高）
Relaistic Vision（更朴素踏实，食物，动物照片，假新闻照片）
L.O.F.I（精致的照片级人像专精模型，面部处理胜过其他，东亚审美也支持）
#2.5D（三维动画）
3D,render,chibi,digital art,concept art,{realistic}
Never Ending Dream (NED)（表现最好，结合lora进行动漫二次创作，真实感恰到好处的满足二次元想象，真实世界里不会产生过分陌生感）
Protogen x3.4(Photorealism)（更接近真实系模型，贴近真实的魔幻感超现实画面）
国风3 (Guofeng3)（符合国人审美，结合其他lora能产生水墨风小人书等风格）
#其他
富有魔幻感的场景Cheese Daddy's Landscapes mix
富有现代感的建筑dvArch-Multi-Prompt Architecture Tuned Model
富有高级感的平面设计Graphic design_2.0

#Prompt:
1girl, detailed background filled with(many:1.1) (colorful:1.1) (flowers):1.1,(quality:1.1), (photorealistic:1.1),(resolution:1.1), (sharpness:1.1),(cinematic lighting), depth of field, Canan EOS R6, 135mm, 1/1250S, f/2.8, ISO 400
white cloth with (lace trim:1.3),close-up,portrait,SFW,
#Negative prompt：
NG_DeepNegative_V1_75T, child, lowres, worst quality, low quality, blurry, fake, 3d, anime, bad anatomy, disabled body,disgusting,ugly, text, watermark,

#Prompt:
SFW,(1girl:1.3),long hair,red hair,face,front,looking at viewer,orange|red dress,upper body,standing,outdoor,Chinese traditional clothes,palace,
(masterpiece:1,2),best quality,masterpiece,highres,original,extremely detailed wallpaper,perfect lighting,(extremely detailed CG:1.2), drawing, paintbrush,
#Negative prompt：
NSFW,(worst quality:2),(low quality:2),(normal quality:2),lowres,normal quality,((monochrome)),((grayscale)),skin spots,acnes,skin blemishes,age spot,(ugly: 1.331),(dupliute:1.331),(mortid:1.21),(mutilated:1.31),(tranny:1.331),mutated hands,(poorly drawn hands:1.5),blurry,(bad anatomy:1.21),(bad proportions:1.331),extra limbs,(disfigured: 1.331),(missing arms: 1.331),(extra legs:1.331),(fused fingers:1.61051),(too many fingers:1,61051),(unclear eyes: 1.331),lowers,bad hands,missing fingers,extra digit,bad hands,missing fingers,(((extra arms and legs))),
CFG scale=8
denoising=0.5
seed=2134139580

#显存消耗(常驻消耗2.3G)
noscript：8G
script：3G
extras: 不消耗显存


#安装wd14-tagger
file=extensions/stable-diffusion-webui-wd14-tagger/tagger/ui.py
cp ${file} ${file}.bk
$SED -i 's@from webui import wrap_gradio_gpu_call@#from webui import wrap_gradio_gpu_call@g' ${file}
$SED -i '/#from webui import wrap_gradio_gpu_call/a\from modules.call_queue import wrap_gradio_gpu_call' ${file}

file=extensions/stable-diffusion-webui-wd14-tagger/preload.py
cp ${file} ${file}.bk
$SED -i 's@from modules.shared import models_path@from modules import paths@g' ${file}
$SED -i "s@default_ddp_path = Path(models_path, 'deepdanbooru')@default_ddp_path = Path(paths.models_path, 'deepdanbooru')@g" ${file}

#tagger插件会启动下载SmilingWolf/wd-v1-4-vit-tagger-v2到hf cache目录，如果下载有问题需要手工修改
mkdir /workspace/hfcache/hub/models--SmilingWolf--wd-v1-4-vit-tagger-v2
#手工下载model.onnx，selected_tags.csv
file=extensions/stable-diffusion-webui-wd14-tagger/tagger/interrogator.py
把以下内容：
model_path = Path(hf_hub_download(
    **self.kwargs, filename=self.model_path))
tags_path = Path(hf_hub_download(
    **self.kwargs, filename=self.tags_path))
修改为：
model_path = "/.cache/huggingface/hub/models--SmilingWolf--wd-v1-4-vit-tagger-v2/model.onnx"
tags_path = "/.cache/huggingface/hub/models--SmilingWolf--wd-v1-4-vit-tagger-v2/selected_tags.csv"

#Prompt:
SFW, masterpiece, best quality, 
1girl,dynamic pose, city background,
#add promtps from wdtagger
#add prompts for charturner(cant' be from wdtagger with a charturnner pic, can't tell the scenerio and purpose)
(easynegative:1.2),A character turnaround of a (corneo_dva) wearing blue mecha bodysuit, (CharTurnerV2:1.2),(multiple views of the same character with the same clothes:1.2), ((character sheet)), (model sheet),((turnaround)),(reference sheet), white background, simple background, character concept, full body,

<lora:dVaOverwatch_v3:0.8>替换(corneo_dva)

DEEVA \(OVERWATCH 1 VERSIONS,D.VA /(OVERWATCH 1/),
DEEVA \(OVERWATCH 2 VERSIONS\),D.VA /(OVERWATCH 2/),
DEEVA \(OVERWATCH BLACK CAT VERSIONS\), D.VA/(OVERWATCH BLACK CAT/),

#Negative prompt：
NSFW, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (ugly:1.331), (duplicate:1.331), (morbid:1.21),(mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured: 1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), lowers, bad hands, missing fingers, extra digit,bad hands, missing fingers, (((extra arms and legs))),

#局部重绘，原来text2img 20步生成的人像发到img2img，人睁眼到闭眼，只有20步时，蒙版的区域色调和其他区域不一样，换mask content没用，50步以后没有明显色差
#如果重绘非mask区域，步数调整到50步还是有色差，调其他的“Only masked padding, pixels”，“CFG Scale”，“Denoising strength”都没用
#“Mask blur”调到12能看上去自然一些。
  fill填充，
  origin原图，
  latent noise潜变量噪声
  latent nothing潜变量数值零，

inpaint sketch的调色板，safari没有，edge能调出来。blue设置到5比较合适，太高黑口罩颜色跑到脸上了。

(blue face mask with 1 white heart sign:2)
直接text2img无法生成，重绘口罩加心形，最好就是多个浅蓝色卫生口罩的。
image sketch心形有时容易生成，有时怎样也调不出来。最后一次调出来：
  cfg scale: 8
  denoising strength: 0.4
  prompt height:1.8
发现咯吱窝很难看inpaint/inpaint sketch都很难美化，加了提示词有反效果，咯吱窝奇怪的东西更多了。

paint upload
狗坐凳子换成老虎坐凳子，
1，先把狗坐凳子的tag提取出来。
2，调节参数
  cfg scale: 14
  denoising strength: 0.5
  prompt height:1.2
3，老虎的眼睛很模糊，把Masked content从original改成fill以后，老虎变成背对镜头，最前面加上front的提示词。

给tag_complete增加中文和翻译词库文件，并不能把中文提示词翻译过来
换成用另一个插件all-inone翻译插件，补全又不工作了。

cutoff不起作用，删除掉dva的lora <>和专用提示词也没法出现黄色头盔和红色手套，还不如在提示词里多加权重。

lucy_cyberpunk这个LORA，和网站说的底模1.5搭配，出来的图像鬼一样丑，和深渊橘就很符合。

Fashion Girl，时尚女性
Cute Girl，mix4,
Asian Male
吉卜力，ghibli style, howl \(howl no ugoku shiro\)
princess zelda，princess zelda
gacha splash style(dont use SDE), [(white background:1.5)::5], isometric OR hexagon, 1girl, midshot, full body, hires fix + SD upscale, MultiDiffusion
  magician,blue long dress,jewelry,holding a wand,
  ocean,sea waves,water splashes,sky,light particles,VFX,night,starry sky,galaxy,magic power,water drop
anime tarot card art style
zyd232's statis pod/chamber
mugshot
lottalewds' thisisfine
Mecha, 
  best quality ,masterpiece, illustration, an extremely delicate and beautiful, extremely detailed ,CG ,unity ,8k wallpaper, Amazing, finely detail, masterpiece,best quality,official art,extremely detailed CG unity 8k wallpaper,absurdres, incredibly absurdres, ultra-detailed, highres, extremely detailed,beautiful detailed girl,light on face,
  mecha, armor, mechanical_body, spaceship, city, cyberpunk, star_sky
  cyberpunk,futuristic,intricate mechanical bodysuit, mecha corset,mechanical parts,robostic arms and legs,headgear,caustics,reflection,ray tracing,demontheme,cyber effect,science fiction



  a futuristic looking cyborg girl with a black cyberhelmet head with red triangle led lights and a halo

给mecha girl换一个cyber helmet头盔总是一团亮没有细节。只保留头盔提示词，只重绘头部，稍微好点，但还是太亮。  

pip install insightface
pip install xformers
cd extensions
git clone https://github.com/Mikubill/sd-webui-controlnet.git
#把hf-mirror搜索lllyasviel/ControlNet-v1-1下载的yaml和pth文件，放到extensions/sd-webui-controlnet/models文件夹里
file=extensions/sd-webui-controlnet/scripts/api.py
cp ${file} ${file}.bk
$SED -i 's@from typing import Union@from typing import Union, List@g' ${file}

prompt:
(sfw:1.2),absurdres,1girl,ocean,white dress,long sleeves,sunhat,smile,
negative prompt:
nsfw,(worst quality:1.2),(low quality:1.2),(lowres:1.1),(monochrome:1.1),(greyscale),multiple views,comic,sketch,animal ears,pointy ears,blurry,transparent,see-through

best quality, masterpiece, (photorealistic:1.4), 1girl, light smile, shirt with collars, waist up, dramatic lighting, from below
nsfw, ng_deepnegative_v1_75t,badhandv4, (worst quality:2), (low quality:2), (normal quality:2), lowres,watermark, monochrome
euler
7
30
3698311310
高清修复1分钟

masterpiece, best quality,
1girl, black skirt, branch, building, chain-link fence, cherry blossoms, fence, long hair, outdoors, petals, pleated skirt, rain, shirt, short sleeves, skirt, solo, standing, tree,
sky,street
negativePrompt:badhandv4, EasyNegative, verybadimagenegative_v1.3,illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, normal quality, ((monochrome)), ((grayscale))
Mask blur:"4"
SD upscale upscaler:"R-ESRGAN 4x+"
Denoising strength:"0.2"
Clip skip:"2"
seed:1173118963
SD upscale overlap:"64"
Size:"2048x1152"
Model hash:"2bc86c5322"
resources:[]
steps:50
sampler:"Euler a"
Eta:"0.67"
cfgScale:7

stone,rock,coal,sand,
golden,gold,glid,
from above,
stone,rock,coal,sand,
high quality,highres，masterpiece,solid background

(worst quality:2),(low quality:2),(normal quality:2),lowres,watermark,monochrome

best quality,masterpiece,(photorealistic:1.4),1girl,dramatic lighting,full body,indoors,dappled sunlight,sunlight,clothed
nsfw,ng_deepnegative_v1_75t,badhandv4,(worst quality:2),(low quality:2),(normal quality:2),lowres,watermark,monochrome

brightness/illumination/qrcode不知道怎么配置json/yaml，一直无法实现相应效果

1girl, nilou (genshin impact), back tattoo, solo, horns, long hair, red hair, sky, looking at viewer, puffy long sleeves, long sleeves, veil, cloud, vision (genshin impact), bangs, skirt, thighlet, blue eyes, tattoo, blue sky, floating hair, twintails, fake horns, puffy sleeves, breasts, outdoors, harem outfit, water, blush, looking back, blue skirt, parted bangs, very long hair, smile, day, back, white headwear, bracer, jewelry, gold trim, closed mouth
NSFW, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (ugly:1.331), (duplicate:1.331), (morbid:1.21),(mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured: 1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), lowers, bad hands, missing fingers, extra digit,bad hands, missing fingers, (((extra arms and legs))),

git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-rembg
ownloading data from 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx' to file '/workspace/shouxiesd/stable-diffusion-webui-master/models/u2net/isnet-anime.onnx'
:<<EOF
pip uninstall onnxruntime-gpu
pip install https://nvidia.box.com/shared/static/qnm7xtdemybuyog3yzz4qio3ly8fvi6r.whl
16.1的onnxruntime不能兼容cuda 12，降到11.8可以运行
EOF
EOF

git clone git@github.com:Bing-su/adetailer.git
#<lora:lnstantPhotoX3:.5>,
1giri, (photorealistic:1.4), realistic, masterpiece, best quality, full body, standing, red blouse, light and shadow, perfect light, indoors, backlighting, 
<lora:FilmVelvia3:.5>
nsfw, ng_deepnegative_v1_75t,badhandv4, (worst quality:2), (low quality:2), (normal quality.2), lowres,watermark, monochrome,

detailed face,close-up,portrait


(photorealistic:1.4), realistic, masterpiece, best quality, light and shadow, perfect light, outdoors,
8k raw photo,bokeh,depth of field,professional,4k,detailed face,
<lora:FilmVelvia3:.5>,
weapon, gun, rifle, helmet, tree, multiple boys, hat, holding, holding weapon

nsfw, ng_deepnegative_v1_75t,badhandv4, (worst quality:2), (low quality:2), (normal quality.2), lowres,watermark, monochrome,


git clone git@github.com:pkuliyi2015/sd-webui-stablesr
1girl, outdoor, athletes, running, jumping, gym, Olympics, hurdling,
(masterpiece:2), (best quality:2), (realistic:2),(very clear:2),
3d, cartoon, anime, sketches, (worst quality:2), (low quality:2),

masterpiece, best quality, (photorealistk:1.3), 8k raw photo, bokeh, depth of field, professional, 4k, highly detailed, detailed face, realistic,.35mm photograph, professional, 4k, highly detailed,
1boy, asian, Chinese, athlete, shooter, weapon, gun, male focus, solo, watch, handgun, jacket, wristwatch, holding weapon, pulling the trigger, holdinggun with one hand, Airsoft gun, Olympic Games, shooter, bullet box, shooting range, aiming, outdoors, gym,
(red clothes:1.3),
drawing, painting crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers,
extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (monochrome,greyscale, old photo:1.5), high contrast,

1girl,jewelry, solo, realistic, earrings, closed mouth, hat, portrait, female focus
3d, cartoon, anime, sketches, (worst quality:2), (low quality:2),

黑白照片上色的controlnet recolor如果正向提示词完全无法上色，去掉正向提示词就没问题了。

下载到extensions文件夹后，启动到Commit hash:后，卡在这里不动很久都没进入listen状态，
启动脚本加上参数 --loglevel=DEBUG，发现停在

#git clone git@github.com:NVIDIA/Stable-Diffusion-Webui-TensorRT
git clone -b dev --single-branch git@github.com:NVIDIA/Stable-Diffusion-Webui-TensorRT
#git clone git@github.com:andrewtvuong/Stable-Diffusion-WebUI-TensorRT
TensorRT扩展插件安装的BUG已经修复，但没有发布在主分支中。所以可以切换到dev分支，就可以顺利安装，也可以手动安装 https://github.com/andrewtvuong/Stable-Diffusion-WebUI-TensorRT
DEBUG [root] Installing Stable-Diffusion-Webui-TensorRT
DEBUG [PIL.Image] Image: failed to import FpxImagePlugin: No module named 'olefile'
DEBUG [PIL.Image] Image: failed to import MicImagePlugin: No module named 'olefile'
tmndMix: 3.9sec
tmndMix(TRT unet): 1.6sec，需要预热一次
tmndMix-xformers: 2.9
sdxl-base: 6.6
sdxl-base: 5.9，提示“Warning Enabling PyTorch fallback as no engine was found”。用固定512,75token降低到2.8，但是token数目不够生成的图像不一样了
sdxl-base-xformers: 5.6
sdxl，生成512fix和768-1024dynamic 2个TRT后再测试
512：5.9，好像并没有自动选择fix的
768：12.5
1024：10.1爆显存了
512-hires.fix-1024：30.4

masterpiece, best quality, (photorealistk:1.3), 8k raw photo, bokeh, depth of field, professional, 4k, highly detailed, detailed face, realistic,.35mm photograph, professional, 4k, highly detailed,
1boy, asian, Chinese, athlete, shooter, weapon, gun, male focus, solo, watch, handgun, jacket, wristwatch, holding weapon, pulling the trigger, holdinggun with one hand, Airsoft gun, Olympic Games, shooter, bullet box, shooting range, aiming, outdoors, gym,
(red clothes:1.3),
<lora:ganyu_ned2_offset:1> 

drawing, painting crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers,
extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (monochrome,greyscale, old photo:1.5), high contrast,

#lora的tft，不用git clone -b lora_v2好像也可以生成一个TRT多个LORA，但是有时显示第一个有时全部3个都显示有bug。
loraTRT以前加上lora prompt和生成以后，速度差不多，同样随机数下图像也一样。
原模型：5.1，男枪手
原模型，加ganyu_ned2_offset lora prompt：5.5，女枪手
TRT模型，无LORA TFT，加ganyu_ned2_offset lora prompt：4.2，男枪手
TRT模型，LORA TFT，加ganyu_ned2_offset lora prompt：
  没有重新启动还是男枪手，重新启动以后不用加lora prompt随机生成多张全部是中性化的抢手，lora已经和底模合在一起了
  加上lora Prompt还是男性化，只是随机种子不变但是画面变了好像更写实好看了一点，2.1S
  加上lora TRT，加上lora Prompt，画面和上面没有LORATRT一样，1.7S
  再多生成一个lucy的LORA，结论：好像是支持多个LORA，profile的显示也一直显示多个
    ganyu的prompt下画面不变，
    lucy的prompt下变成女的

没有lora_v2这个分支，clone失败

-b dev NVIDIA这个分支
  生成图片和非TRT不一样了
  支持多个LORA

yum install xdg-utils -y
git clone git@github.com:aigc-apps/sd-webui-EasyPhoto
DEBUG [root] Installing sd-webui-EasyPhoto
is_installed check for tensorflow-cpu failed as 'spec is None'
is_installed check for modelscope failed as 'spec is None'
is_installed check for av failed as 'spec is None'
is_installed check for diffusers failed as 'spec is None'
is_installed check for nvitop failed as 'spec is None'
选了其他模型，还是会下载这个Start Downloading: https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/ChilloutMix-ni-fp16.safetensors
pip3 install --force-reinstall nvidia-ml-py nvitop
ZacLiu8picsWTBG
chilloutmix,训练出错：
[rank0]:     info = text_model.load_state_dict(converted_text_encoder_checkpoint, strict=False)
[rank0]:   File "/data0/envs/sd-webui-master/lib/python3.10/site-packages/torch/nn/modules/module.py", line 2189, in load_state_dict
[rank0]:     raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
[rank0]: RuntimeError: Error(s) in loading state_dict for CLIPTextModel:
[rank0]:        size mismatch for text_model.embeddings.token_embedding.weight: copying a param with shape torch.Size([49408, 768]) from checkpoint, the shape in current model is torch.Size(

docker run -it --network host --gpus all registry.cn-shanghai.aliyuncs.com/pai-ai-test/eas-service:easyphoto-diffusers-py310-torch201-cu117
image太大，可能硬盘撑爆放弃。

rembg以后生成的图片
  有些会生成jpg/png两组，有些只生成png，
  png是4通道的，做facal crop时会失败，
  把png复制保存一份，如果有jpg，png删掉

SDXL
1 18 yo beautiful asian girl walking in the forest,photographic
SDXL 1 18 yo 生成的都是小女孩，改成One eighteen years old就可以正常生成，更需要自然语言而不是训练时的咒语

加上手部的prompt以后，手生成畸形，
加上负面提示词没用，
加上ADetailer对手处理也没用，
单加negativeXL_D没用，加上(negativeXL_D:1.2)以后手脚不多了，但是胳膊细长畸形，
提示词换成自然语言风格也没用：A photographic close-up picture of one eighteen years old beautiful asian girl walking in the forest,looking at viewer, with hands up,
同样的随机数，降低分辨率到768就没事，到800就会又乱了
One eighteen years old beautiful asian girl walking in the forest,photographic,looking at viewer,close-up,hands up,

drawing, painting crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, lowres, bad anatomy, bad hands, text, error, missing fingers,
extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, (monochrome,greyscale, old photo:1.5), high contrast,
NSFW, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (ugly:1.331), (duplicate:1.331), (morbid:1.21),(mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured: 1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), lowers, bad hands, missing fingers, extra digit,bad hands, missing fingers, (((extra arms and legs))),

#git clone git@github.com:wcde/sd-webui-refiner
效果和只用base差不多，好像没有调用refiner一样，两部分开同样的种子refiner对构图尤其脸部又较大改变
对refiner阶段的参数，只有一个step参数可调

git clone git@github.com:ModelSurge/sd-webui-comfyui.git
cd sd-webui-comfyui
git clone git@github.com:comfyanonymous/ComfyUI

修改settting/comfyUI里的listen为0.0.0.0
把Installation里的安装位置从/workspace/shouxiesd/stable-diffusion-webui-master/extensions/sd-webui-comfyui/ComfyUI
改到/workspace/shouxiesd/ComfyUI，可以复用独立安装里的插件

pip install pillow_heif

git clone git@github.com:continue-revolution/sd-webui-animatediff 
rm -rf extensions/sd-webui-animatediff/model
ln -s /workspace/shouxiesd/animatediff_models /workspace/shouxiesd/stable-diffusion-webui-master/extensions/sd-webui-animatediff/model
rm -rf models/Lora/animatediff
ln -s /workspace/shouxiesd/animatediff_lora /workspace/shouxiesd/stable-diffusion-webui-master/models/Lora/animatediff
git clone git@github.com:deforum-art/sd-webui-deforum

(masterpiece, best quality), 1girl, wavy hair, looking at viewer, blurry foreground, upper body, necklace, contemporary, plain pants, ponytail, freckles, red hair, dappled sunlight, smile, happy,
(worst quality, low quality, letterboxed), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature,watermark, username, blurry

upper body, an image of an anime girl with bright red hair, wearing a sundress and holding a bouquet of wildflowers, standing in a field of tall grass with a soft breeze blowing through, with a sense of peace and tranquility in the air, fluttering skirts, moving clouds, flying petals
(badhandv4:1.2), (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, badhands, ((monochrome)), ((grayscale)), watermark, moles, large breast, big breast,

1girl, solo, high heels, skirt, long hair, black hair, outdoors, full body, shoes, standing, looking at viewer,the girl is dancing,<lora:ganyu_ned2_offset:1> 

best quality, masterpiece, 1girl, upper body, detailed face, looking at viewer, outdoors, upper body, standing, outdoors,
0: closed eyes,(spring:1.2), cherry blossoms, falling petals, pink theme,
16: open eyes,(summer:1.2), sun flowers, hot summer, green theme
32: closed eyes,(autumn:1.2), falling leaves, maple leaf, yellow trees, orange theme 
48: open eyes,(winter.1.2), snowing, snowflakes, white theme

sdxl的3个animatediff 2个safesensorts,1个ckpt，sdxl除了base还换了几个其他的，都提示
params.prompt_scheduler.save_infotext_txt(res)
    AttributeError: 'NoneType' object has no attribute 'save_infotext_txt'

A cup of coffee steaming on the table
A small boat was sailing on the stormy sea
A snowy mountain with aurora in the sky
masterpiece, best quality
(worst quality, low quality, letterboxed), lowres,
Zoom: (1.0025+0.002*sin(1.25*3.14*t/30))->(1.03)
Translation X: ->,60:(10),90:(0)
Max Frames: 120->90

   "0": "no_humans, cyberpunk, futuristic, sci-fi, outdoors, scenery, futuristic city, sunset, afternoon, lens flare, from below, wide angle shot -neg (lgirl:1.2), people",
   "30": "no_humans, city skylines, city horizon, cloudy skies, planes, stars, the moon, night, night sky, cityscape, city horizon, comets, meteors, meteor showers --neg 1girl, people",
   "60": "no_humans, standing on mountain peak, misty, epic, fantasy, desolate, galaxy in the background,huge moon on the horizon, from above, comets, meteors, meteor showers-neg Igiri, people",
   "90": "universe, galaxy, milky ways, earth from space, Planet earth from the space at night, planets,dome, lift up, sunrise"


Translation Y: 0:(0.2*t)

Translation X:0:(0),10:(-4),90:(-4.5),120(0)
Translation Y:0:(0)
Translation Z:0:(0),60:(1),90:(2),120:(0)
3D rotation X:0:(0),90:(0),120:(2)
3D rotation Y:0:(0),10:(1),100:(1),120:(0)
3D rotation Z:0:0

     "0": "green lemon, food, fruit, no humans, food focus, simple background, still life, realistic, black background, green theme",
     "25": "no humans, green leaves with water drop on the surface, green theme, rain, still life",
     "50": "nature, forest, scenery, no humans, outdoors, tree, sunlight, day, path, road, grass, plant, green theme",
     "75": "scenery, sunset, cloud, sky, sun, outdoors, horizon, ocean, cloudy sky, water, lens flare, reflection,sunlight, dutch angle, beach",
     "100": "candle, burning candle in a dark room, glowing, flame, warm, black background"

rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
yum install ffmpeg ffmpeg-devel -y
git clone git@github.com:s9roll7/ebsynth_utility.git 

1boy,(masterpiece:1.2), best quality, masterpiece, hires, original, dynamic pose, detailed face, illustration, anime, street, outdoor, stairs, dancing, orange coat,yellow waistcoat, suit, joker,cinematic light, perfect light.
NSFW, (worst quality:2), (low quality:2), (normal quality:2), 3d, render, lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny: 1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions: 1.331), extra limbs, (disfigured: 1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), lowers, bad hands, missing fingers, extra digit,bad hands, missing fingers, (((extra arms and legs))),
ebsynth没有linux版本，第5步生成2个项目文件以后要在mac上处理

git clone git@github.com:Scholar01/sd-webui-mov2mov
#ImportError: cannot import name 'create_sampler_and_steps_selection' from 'modules.ui'
file=modules/ui.py
cp ${file} ${file}.bk
$SED -i '/if not cmd_opts.share and not cmd_opts.listen/i\' ${file}
def create_sampler_and_steps_selection(choices, tabname):
    return scripts.scripts_txt2img.script('Sampler').steps, scripts.scripts_txt2img.script('Sampler').sampler_name

squatting, petting, shirt, outdoors, black hair, cat, shoes, white shirt, solo, sneakers, rabbit, short hair, from side, day, 1boy, profile, pants, short sleeves,boy touch rabbit
(badhandv4:1.2), (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, badhands, ((monochrome)), ((grayscale)), watermark, moles, large breast, big breast,

git clone git@github.com:WSH032/kohya-config-webui.git
file=modules/interrogate.py
cp ${file} ${file}.bk
$SED -i 's@model_base_caption_capfilt_large.pth@model_base_capfilt_large.pth@g' ${file}

git clone git@github.com:SpenserCai/sd-webui-deoldify

wget -c https://github.com/TimDettmers/bitsandbytes/releases/download/0.43.0/bitsandbytes-0.43.0-py3-none-manylinux_2_24_x86_64.whl
pip debug --verbose|grep py310-none-manylinux
mv bitsandbytes-0.43.0-py3-none-manylinux_2_24_x86_64.whl bitsandbytes-0.43.0-py3-none-manylinux_2_17_x86_64.whl
pip install bitsandbytes-0.43.0-py3-none-manylinux_2_17_x86_64.whl
git clone git@github.com:d8ahazard/sd_dreambooth_extension

git clone git@github.com:Tencent/LightDiffusionFlow.git

git clone git@github.com:0xbitches/sd-webui-lcm

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
export LD_PRELOAD=/lib64/libtcmalloc.so.4
yum install -y libaio-devel
echo 'export CFLAGS="\$CFLAGS -I/usr/include" && export LDFLAGS="\$LDFLAGS -L/usr/lib64"' >> ~/.bashrc
#设置错了CFLAGS，改正后错误值留存在vscode的配置中
#rm -rf ~/.vscode-server
#最新版本deepspeed 不能用zero3训练，报错：RuntimeError: 'weight' must be 2-D
#https://github.com/arcee-ai/DistillKit/issues/3
pip install diffusers[training] deepspeed==0.14.5 torchvision huggingface_hub==0.25.2
conda install cuda-toolkit
cp /workspace/shouxiellm/qwen/Qwen2/examples/sft/ds_config_zero3.json /workspace/shouxiesd/ds_config_zero3.json
:<<EOF
从ds_config_zero3.json中删除optimizer因为
unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
result = self._prepare_deepspeed(*args)
ValueError: You cannot specify an optimizer in the config file and in the code at the same time. 
EOF
accelerate config
:<<EOF
No distributed training
CPU only: NO
torch dynamo: NO
DeepSpeed: yes
  /workspace/shouxiesd/ds_config_zero3.json
GPU: 0
#FP16 or BF16: NO

Do you want to enable Mixture-of-Experts training (MoE)? [yes/NO]:NO

删除
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
因为
ValueError: Can find neither `model.config.hidden_size` nor `model.config.hidden_sizes`, therefore it's not possible to automatically fill out the following `auto` entries in the DeepSpeed config file

EOF
file=/workspace/hfcache/accelerate/default_config.yaml
cp ${file} ${file}.bk
\cp /workspace/shouxiesd/accelerrate_config.yaml /workspace/hfcache/accelerate/default_config.yaml
#num_processes必须是1，不然GPU会从0增长
#install cutlass
git config --global credential.helper store
huggingface-cli login
  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
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
accelerate launch text_to_image/train_text_to_image_lora.py --mixed_precision="fp16" \
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
export INSTANCE_DIR="${SD_HOME}/dog-example"
#export INSTANCE_DIR="${SD_HOME}/zacliu-example"
rm -rf ${INSTANCE_DIR}/.huggingface
export OUTPUT_DIR="${SD_HOME}/output_dreambooth_lora_dog"
#export OUTPUT_DIR="${SD_HOME}/output_dreambooth_lora_zacliu"
#提示保存deepspeed checkpoint失败
#修改accelerate config文件，去掉FP16混合精度，不用deepspeed
#提示ValueError: Attempting to unscale FP16 gradients
#升级peft没用，修改accelerate config文件，去掉FP16混合精度
pip install --upgrade peft
#  --instance_prompt="a photo of zacliu boy" \
#  --validation_prompt="A photo of zacliu boy in a bucket" \
accelerate launch dreambooth/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
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

 #https://blog.csdn.net/m0_46864820/article/details/137234801
 git clone git@github.com:lllyasviel/ControlNet orig-ControlNet
 ControlNet
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

while ! git clone https://github.com/comfyanonymous/ComfyUI_examples.git; do sleep 2 ; done ; echo succeed
while ! git clone https://github.com/comfyanonymous/ComfyUI.git; do sleep 2 ; done ; echo succeed
cd ComfyUI
pip install -r requirements.txt

pip install onnxruntime onnx
pip install timm

cp extra_model_paths.yaml.example extra_model_paths.yaml
$SED -i 's@    base_path: path/to/stable-diffusion-webui/@    base_path: /workspace/shouxiesd/stable-diffusion-webui-master/@g' extra_model_paths.yaml
$SED -i 's@    controlnet: models/ControlNet@    controlnet: /workspace/shouxiesd/stable-diffusion-webui-master/extensions/sd-webui-controlnet/models@g' extra_model_paths.yaml

HF_ENDPOINT=https://hf-mirror.com python main.py --listen 0.0.0.0

prompt: 
1girl, school uniform, purple theme, galaxy, universe, cinematic lighting, perfect lights, vara lights, masterpiece, best quality,
negative prompt:
nsfw, (low quality, worst quality:1.5),lowres, bad anatony, bad hands, text, error, missing fingers, extra digit, fewer digits, 
cropped, worst quality, low quality, normal quality, jpeg artifacts signature, watermark, usernane, blurry

cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager
git clone https://github.com/11cafe/comfyui-workspace-manager
git clone git@github.com:atmaranto/ComfyUI-SaveAsScript
git clone git@github.com:pythongosssss/ComfyUI-WD14-Tagger
pip install onnxruntime
git clone https://github.com/AIrjen/OneButtonPrompt
git clone git@github.com:ltdrdata/ComfyUI-Inspire-Pack.git
git clone git@github.com:jags111/efficiency-nodes-comfyui.git
pip install simpleeval
git clone git@github.com:Suzie1/ComfyUI_Comfyroll_CustomNodes.git
pip install numba
git clone git@github.com:WASasquatch/was-node-suite-comfyui.git
git clone git@github.com:rgthree/rgthree-comfy.git
git clone git@github.com:Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
git clone git@github.com:cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/Gourieff/comfyui-reactor-node
git clone git@github.com:ssitu/ComfyUI_UltimateSDUpscale --recursive
pip install -r requirements.txt
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
pip install -r requirements.txt
git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui
git clone git@github.com:twri/sdxl_prompt_styler.git

界面restart不行，需要手工重启main.py，语言和提示词补全插件设置菜单才会出现。
AIGODLIKE-COMFYUI-TRANSLATION 
ComfyUI-Manager
ComfyUI_Custom_Nodes_AlekPet
ComfyUI-SaveAsScript
ComfyUI-Custom-Scripts
comfyui-workspace-manager
ComfyUI-WD14-Tagger
OneButtonPrompt
ComfyUI-Inspire-Pack
efficiency-nodes-comfyui
ComfyUI_Comfyroll_CustomNodes
was-node-suite-comfyui
rgthree-comfy
ComfyUI-AnimateDiff-Evolved
ComfyUI_IPAdapter_plus
comfyui-reactor-node
ComfyUI_UltimateSDUpscale
masquerade-nodes-comfyui
sdxl_prompt_styler

git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
cd ComfyUI-Impact-Pack
git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack impact_subpack
#python install.py


A hyper-realistic portrait of a 20 yo girl standing in a wheat field
with her back to the camera, cloudy sky. sunrise, depth of field, and
blurred background

git clone git@github.com:Kosinkadink/ComfyUI-VideoHelperSuite
git clone git@github.com:cubiq/ComfyUI_essentials

git clone git@github.com:storyicon/comfyui_segment_anything
python install.py
#在install missing nodes里try fixing

git clone git@github.com:Kosinkadink/ComfyUI-AnimateDiff-Evolved.git
rm -rf custom_nodes/ComfyUI-AnimateDiff-Evolved/models
ln -s /workspace/shouxiesd/animatediff_models /workspace/shouxiesd/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models
rm -rf custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora
ln -s /workspace/shouxiesd/animatediff_lora /workspace/shouxiesd/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora

git clone git@github.com:0xbitches/ComfyUI-LCM

git clone git@github.com:TheMistoAI/ComfyUI-Anyline.git

git clone git@github.com:Kosinkadink/ComfyUI-Advanced-ControlNet

HF_MODEL_NAME = "lllyasviel/Annotators",
DWPOSE_MODEL_NAME = "yzd-v/DWPose",
BDS_MODEL_NAME = "bdsqlsz/qinglong_controlnet-lllite",
DENSEPOSE_MODEL_NAME = "LayerNorm/DensePose-TorchScript-with-hint-image",
MESH_GRAPHORMER_MODEL_NAME = "hr16/ControlNet-HandRefiner-pruned",
SAM_MODEL_NAME = "dhkim2810/MobileSAM",
UNIMATCH_MODEL_NAME = "hr16/Unimatch",
DEPTH_ANYTHING_MODEL_NAME = "LiheYoung/Depth-Anything", #HF Space
DIFFUSION_EDGE_MODEL_NAME = "hr16/Diffusion-Edge"

/usr/bin/ffmpeg -v error -n -i /workspace/shouxiesd/ComfyUI/output/AnimateDiff_00001.mp4 -ar 44100 -ac 2 -f f32le -i - -c:v copy -c:a aac -af apad=whole_dur=36.6875 -shortest /workspace/shouxiesd/ComfyUI/output/AnimateDiff_00001-audio.mp4

git clone git@github.com:Fannovel16/ComfyUI-Frame-Interpolation

git clone git@github.com:tinyterra/ComfyUI_tinyterraNodes

1girl, beautiful, Southeast Asian face and skin, sitting by a desk, looking at viewer,facing the camera,close-up,upper body, talking to viewer,
masterpiece,best quality,cinematic lighting,HDR,UHD,8K,

embedding:bad_prompt_version2-neg, embedding:easynegative, embedding:negative_hand-neg, embedding:ng_deepnegative_v1_75t, 

masterpiece,best quality,cinematic lighting,HDR,UHD,8K,

git clone git@github.com:kijai/ComfyUI-KJNodes.git
git clone git@github.com:djbielejeski/a-person-mask-generator

pip install rembg
pip install insightface

######################end of ComfyUI######################

######################start of stable-diffusion-webui-forge######################
conda create -n sd-webui-forge python=3.10 -y
conda activate sd-webui-forge77 

cd stable-diffusion-webui-forge

pip install deepspeed
#cp -r ../stable-diffusion-webui-master/repositories/generative-models repositories/
#mkdir stable-diffusion-webui-forge
#rsync -a --exclude=outputs --exclude=models --exclude=extensions --exclude=embeddings stable-diffusion-webui-master/ stable-diffusion-webui-forge/
git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git -o stable-diffusion-webui-forge
git config --global --add safe.directory /workspace/shouxiellm/sd/stable-diffusion-webui-forge
pip install -r requirements_versions.txt
#git clone https://github.com/Stability-AI/generative-models -o repositories/generative-models
file=webui.sh
cp ${file} ${file}.bk
#增加set -e，循环安装避免git clone出错
file=webui-user.sh
cp ${file} ${file}.bk
\cp ../webui-user.sh ./
$SED -i 's@stable-diffusion-webui-master@stable-diffusion-webui-forge@g' ${file}
while ! bash webui.sh -f; do sleep 2 ; done ; echo succeed

mv embeddings embeddings.bk
ln -s /workspace/shouxiesd/stable-diffusion-webui-master/embeddings /workspace/shouxiesd/stable-diffusion-webui-forge/embeddings
mv extensions extensions.bk
ln -s /workspace/shouxiesd/stable-diffusion-webui-master/extensions /workspace/shouxiesd/stable-diffusion-webui-forge/extensions
mv models models.bk
ln -s /workspace/shouxiesd/stable-diffusion-webui-master/models /workspace/shouxiesd/stable-diffusion-webui-forge/models
ln -s /workspace/shouxiesd/stable-diffusion-webui-master/outputs /workspace/shouxiesd/stable-diffusion-webui-forge/outputs

git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/a1111-sd-webui-tagcomplete
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/sd-webui-model-downloader-cn
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/stable-diffusion-webui-images-browser
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/infinite-zoom-automatic1111-webui
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/stable-diffusion-webui-wd14-tagger
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/ultimate-upscale-for-automatic1111
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/sd-webui-controlnet
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/sd-webui-llul

HF_ENDPOINT=https://hf-mirror.com python3 launch.py -f --skip-torch-cuda-test --skip-version-check --listen --enable-insecure-extension-access
# --no-half --precision full

cd extensions
git clone git@github.com:layerdiffusion/sd-forge-layerdiffuse
git config --global --add safe.directory /workspace/shouxiesd/stable-diffusion-webui-master/extensions/sd-forge-layerdiffuse'

:<<EOF
git remote -v 
git remote add forge https://github.com/lllyasviel/stable-diffusion-webui-forge
git branch lllyasviel/main
git checkout lllyasviel/main
while ! git fetch forge; do sleep 2 ; done ; echo succeed
git branch -u forge/main
export LD_PRELOAD=/lib64/libtcmalloc.so.4
while ! git pull; do sleep 2 ; done ; echo succeed
unset LD_PRELOAD
EOF

#prompt:
A yellow apple
A girl with messy curvy hair wearing black leather jacket,upper body,half body,high quality,masterpiece,
A transparent glass bottle
A professional 3d model of a old magic book,glowing effect,red fire and light particles,high quality,best quality,masterpiece,
best quality,masterpiece,original,1girl wearing white cloth and black skirt in the classroom,indoors,afternoon,sunset,cinematic lights
A photo of a women wearing yellow dress walking on the street,outdoors,day,cloudy blue sky,
#negative prompt:
worst quality,ugly,
(ugly,worst quality:1.2),
nsfw,ugly,worst quality,
nsfw,worst quality,text,ugly,negativeXL_D

XL的生成透明图效果较差，XL根据fore生成blend后还有透明色块
1.5的


Downloading: "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/ /workspace/shouxiesd/stable-diffusion-webui-forge/models/layer_model/
https://hf-mirror.com/LayerDiffusion/layerdiffusion-v1/tree/main
点击生成会从hf下载一些模型，需要提前下载好到models/layer_model/目录
sdxl的基本功能，这3个就够了：
layer_xl_transparent_attn.safetensors
layer_sd15_transparent_attn.safetensors
vae_transparent_decoder.safetensors
vae_transparent_encoder.safetensors



######################end of stable-diffusion-webui-forge######################

######################end of svd-ft######################
conda create -n svd-ft python=3.10 -y
conda activate svd-ft

git clone git@github.com:soulteary/docker-stable-video-diffusion.git

pip install torch
pip install transformers==4.35.2 gradio==4.13.0 diffusers==0.25.0 accelerate==0.25.0
pip install opencv-fixer==0.2.5
python -c "from opencv_fixer import AutoFix; AutoFix()"

######################end of svd-ft######################

######################start of EasyPhoto######################
conda create -n easyphoto python=3.10 -y
conda activate easyphoto

git clone git@github.com:aigc-apps/EasyPhoto.git
cd EasyPhoto
pip install -r requirements.txt
pip install --upgrade httpx
ln -s /workspace/shouxiesd/Stable-diffusion /workspace/shouxiesd/EasyPhoto/model_data/Stable-diffusion
python app.py --listen --port 7861
和sd-webui一样的CLIPText模型的权重加载问题
下载的模型和其他文件太大，硬盘撑爆放弃

######################end of EasyPhoto######################

######################start of Kohya_ss######################
conda create -n kohya_ss python=3.10 -y
conda activate kohya_ss

git clone --recursive git@github.com:bmaltais/kohya_ss.git
#Change into the kohya_ss directory:
cd kohya_ss
#If you encounter permission issues, make the setup.sh script executable by running the following command:
chmod +x ./setup.sh
#Run the setup script by executing the following command:
yum install -y dnf
dnf install python3-tkinter -y
./setup.sh
Error: Error downloading packages:
  Cannot download Packages/python3-tkinter-3.6.8-21.el7_9.x86_64.rpm: All mirrors were tried
#放弃使用和sd-webui集成的训练插件
######################end of Kohya_ss######################
