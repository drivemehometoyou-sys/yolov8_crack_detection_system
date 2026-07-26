#coding:utf-8
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 图片及视频检测结果保存路径
save_path = os.path.join(BASE_DIR, '结果')

# 使用的模型路径
model_path = os.path.join(BASE_DIR, 'runs', 'detect', 'exp5', 'weights', 'best.pt')



names = {0 : '裂缝'}

CH_names = ['裂缝']