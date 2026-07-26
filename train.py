#coding:utf-8
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 原版训练（已注释） ==========
# model_path = os.path.join(BASE_DIR, 'yolov8m.pt')
# data_path = os.path.join(BASE_DIR, 'ultralytics', 'cfg', 'datasets', 'my_real_cracks.yaml')

# if __name__ == '__main__':
#     model = YOLO(model_path)
#     results = model.train(data=data_path,
#                           epochs=200,
#                           batch=16,
#                           optimizer='AdamW',
#                           lr0=0.001,
#                           device='0',
#                           workers=4,
#                           project='runs/detect',
#                           name='exp',
#                           )

# ========== 断点续训 ==========
data_path = os.path.join(BASE_DIR, 'ultralytics', 'cfg', 'datasets', 'my_real_cracks.yaml')
checkpoint_path = os.path.join(BASE_DIR, 'runs', 'detect', 'exp5', 'weights', 'last.pt')

if __name__ == '__main__':
    model = YOLO(checkpoint_path)
    results = model.train(data=data_path,
                          epochs=200,
                          batch=16,
                          optimizer='AdamW',
                          lr0=0.001,
                          device='0',
                          workers=4,
                          project='runs/detect',
                          name='exp5',
                          exist_ok=True,
                          resume=True,
                          )








