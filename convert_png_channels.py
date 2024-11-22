from PIL import Image

import cv2
import glob
import os

import argparse
import shutil

from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()
import numpy as np
import imageio

from platform import system
from subprocess import call

def mkdir(path, rm_ifexists=False):
  folder = os.path.exists(path)
  if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
      os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
      print("--- create new folder...  ---:", path)
  else:
      print("---  There is this folder!  ---:", path)
      if rm_ifexists:
        shutil.rmtree(path)
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("--- remove and recreate new folder...  ---:", path)

def livp2Img(path):
  abs_path = os.path.abspath(path)
  orig_path = os.getcwd()
  tmp_path = ".tmp_livp"
  mkdir(tmp_path, True)
  os.chdir(tmp_path) 
  if system() == 'Windows':
      try:
          call(['tar', '--version'], stdout=open(os.devnull, 'w'))
      except FileNotFoundError:
          raise Exception('tar not found, please install it')
  else:
      try:
          call(['unzip', '-v'], stdout=open(os.devnull, 'w'))
      except FileNotFoundError:
          raise Exception('unzip not found, please install it')
  os.system(f'tar -xf "{abs_path}"') if system() == 'Windows' else os.system(f'unzip -q "{abs_path}"')
  listaspacchettati=os.listdir()
  image = None
  for e in listaspacchettati:
    print("e:", e)
    if ".heic" in e:
      pilImage = Image.open(e)
      image = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)
      break
  os.chdir(orig_path)
  shutil.rmtree(tmp_path)
  return image

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', help='输入文件夹')
  parser.add_argument('--output', help='输出文件夹')
  args = parser.parse_args()
  print(args)

  # Get all png files under the input folder
  input_imgs_path = args.input
  extensions = ['png','jpg','jpeg','webp','heic','livp']
  extensions = extensions + [ ext.upper() for ext in extensions]
  extensions = [ '*.' + ext for ext in extensions]
  # 使用glob.glob进行匹配
  input_imgs = []
  for pattern in extensions:
      input_imgs.extend(glob.glob(os.path.join(input_imgs_path, pattern)))
  print("input_imgs:",input_imgs)
  output_imgs_path = args.output
  
  mkdir(output_imgs_path)  # 调用函数

  for file_path in input_imgs:
    # get the file_name of image
    # 在windows下使用“\\”，在linux下使用“/”,注意切换
    file_name = os.path.basename(file_path)
    ext_name = file_name.split('.')[-1].lower()
    print("file_path:", file_path)
    print("file_name:", file_name)
    print("ext_name:", ext_name)

    if ext_name == "heic":
      pilImage = Image.open(file_path)
      image = cv2.cvtColor(np.asarray(pilImage), cv2.COLOR_RGB2BGR)  
    elif ext_name == "livp":
      image = livp2Img(file_path)
      if image is None:
         raise Exception('failt to unzip livp, find/covert heic file')
    else:
      image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    shape = image.shape
    print("shape:", shape)

    file_main_name = file_name.split('.')[0]
    if shape[2] == 4:
      file_main_name = file_main_name + "-4Cto3C"
      print("Image", file_name, " is 4-channel")
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
      print("Image", file_name, " is not 4-channel")
    new_file_path = os.path.join(output_imgs_path, file_main_name + ".png")
    print("new_file_path:", new_file_path)
    if new_file_path != file_path:
      if ext_name != "png" or shape[2] == 4:
        cv2.imwrite(new_file_path, image)
      else:
        shutil.copy(file_path, new_file_path)
    
      
