# -*- coding: utf-8 -*-
import sys
import requests
import os

from typing import List

from alibabacloud_imageseg20191230.client import Client as imageseg20191230Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_imageseg20191230 import models as imageseg_20191230_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient

import os
# 获取所有环境变量
env_vars = os.environ

def download(url, filename):
    req = requests.get(url)
    if req.status_code != 200:
        print('DOWNLOAD_ERROR')
        return
    try:
        with open(filename, 'wb') as f:
            # req.content为获取html的内容
            f.write(req.content)
            print('DOWNLOAD_SUCCESS')
    except Exception as e:
        print(e)


class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client(
            access_key_id: str,
            access_key_secret: str,
    ) -> imageseg20191230Client:
        """
        使用AK&SK初始化账号Client
        @param access_key_id:
        @param access_key_secret:
        @return: Client
        @throws Exception
        """
        config = open_api_models.Config(
            # 您的 AccessKey ID,
            access_key_id=access_key_id,
            # 您的 AccessKey Secret,
            access_key_secret=access_key_secret
        )
        # 访问的域名
        config.endpoint = f'imageseg.cn-shanghai.aliyuncs.com'
        config.region_id = 'cn-shanghai'
        return imageseg20191230Client(config)

    @staticmethod
    def main(
            dir
    ) -> None:
        files = os.listdir(dir)
        for name in files:
            portion = os.path.splitext(name)
            newname = os.path.join(dir, portion[0] + "_seg.png")
            fullname = os.path.join(dir, name)
            with open(fullname, 'rb') as f:
                # 此处记录 AccessKey ID  AccessKey Secret
                client = Sample.create_client(env_vars['ALIYUN_ACCESS_KEY_ID'], env_vars['ALIYUN_ACCESS_KEY_SECRET'])
                segment_hdbody_request = imageseg_20191230_models.SegmentHDBodyAdvanceRequest()
                segment_hdbody_request.image_urlobject = f
                runtime = util_models.RuntimeOptions()
                try:
                    # 复制代码运行请自行打印 API 的返回值
                    result = client.segment_hdbody_advance(segment_hdbody_request, runtime)
                    result_url = result.body.data.image_url
                    print(result_url)
                    download(url=result_url, filename=newname)
                except Exception as error:
                    # 如有需要，请打印 error
                    error = UtilClient.assert_as_string(error.message)
                    print(error)


if __name__ == '__main__':
    # 待处理图片存储路径
    file_dir = "./zacliu-example"
    Sample.main(file_dir)

