import torch
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import check_img_size,Profile,non_max_suppression,scale_boxes,xyxy2xywh
from grabscreen import grab_screen,grab_screen_mss
from utils.augmentations import letterbox
from mouse_control import lock
from models.experimental import attempt_load
import cv2
import win32gui
import win32con
import numpy as np
import pynput
import time
from utils import *

x,y=(1920,1080) #屏幕大小
re_x,re_y=(1920,1080) #游戏屏幕大小

show_win=True

device = "cuda:0" if torch.cuda.is_available() else 'cpu'
half = device != 'cpu'

weights = r'E:\Yolo_project\yolov5-master\runs\train\exp6\weights\best.pt'
# weights = r'E:\Yolo_project\yolov5-master\runs\train\apex.pt'
imgsz=640
data='data/mydata.yaml'
conf_thres=0.35
iou_thres=0.05
flag=0
mouse=pynput.mouse.Controller()
keyboard=pynput.keyboard.Controller()
stride=32
pt=True
names={0: 'ct', 1: 'ct_head', 2: 't', 3: 't_head'} #label

lock_mode=False
def on_move(x, y):
    pass

def on_click(x, y, button, pressed):
    global lock_mode
    if pressed and button == button.x2:
        lock_mode= not lock_mode
        print('lock_mode','on'if lock_mode else 'off')

def on_scroll(x, y, dx, dy):
    pass

# Collect events until released
listener = pynput.mouse.Listener(
    on_move=on_move,
    on_click=on_click,
    on_scroll=on_scroll)
listener.start()



while True:

    start_time_image=time.time()

    #img0=grab_screen_win32(region=(0, 0, x, y))

    img0=grab_screen_mss(region=(0, 0, x, y))

    end_time_grab=time.time()
    #print("grab time",(end_time_grab-start_time_image)*1000,"ms")
    #img0=cv2.resize(img0,(re_x,re_y))

    # Load model
    if not flag:
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=half)
        imgsz = check_img_size(imgsz, s=stride)
        flag=1

    img = letterbox(img0, imgsz, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.

    if len(img.shape) == 3:
        img = img[None]  # img = img.unsqueeze(0)
    torch.cuda.synchronize()
    start_time_pred=time.time()
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    torch.cuda.synchronize()
    aims=[]

    # Process predictions
    for i, det in enumerate(pred):
        s=''
        s += '%gx%g ' % img.shape[2:]
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                line = (cls, *xywh)
                aim=('%g ' * len(line)).rstrip()% line
                print(aim)
                aim = aim.split(" ")
                aims.append(aim)

        if len(aims):
            #if(len(aims)>=2):
            if lock_mode:
                start_time_lock=time.time()
                lock(aims, mouse, re_x, re_y)
                end_time_lock=time.time()
                #print("lock time",(end_time_lock-start_time_lock)*1000,"ms")

            for i,det in enumerate(aims):
                _,x_center,y_center,width,height=det
                x_center,y_center=re_x*float(x_center),re_y*float(y_center)
                width,height=re_x*float(width),re_y*float(height)
                top_left=(int(x_center-width/2), int(y_center-height/2))
                bottom_right=(int(x_center+width/2), int(y_center+height/2))

                color=(0,255,0) #RGB
                cv2.rectangle(img0,top_left,bottom_right,color,4)

    if show_win:
        cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('csgo-detect', re_x // 4, re_y // 4)
        cv2.imshow('csgo-detect',img0)

        hwnd=win32gui.FindWindow(None,'csgo-detect')
        CVRECT=cv2.getWindowImageRect('csgo-detect')
        win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    if cv2.waitKey(1) & 0xFF==ord('q'): #input q to shutdown
        cv2.destroyAllWindows()
        break