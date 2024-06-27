import cv2
from grabscreen import grab_screen_mss
import numpy as np
import heapq
import matplotlib.pyplot as plt
import math
from PIL import Image
import time
import pynput #控制键盘鼠标的输入
from pynput import mouse
mouse = pynput.mouse.Controller()
from pynput import keyboard
kb = keyboard.Controller()
from threading import Thread #这里当时打算多线程？ 我忘了是不是我写的了
import torch
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.general import check_img_size,Profile,non_max_suppression,scale_boxes,xyxy2xywh
from grabscreen import grab_screen ,grab_screen_mss#截屏
from utils.augmentations import letterbox
from mouse_control import lock
from models.experimental import attempt_load
import cv2
import win32gui
import win32con #win32 调用的windows的接口
import numpy as np
import pynput #控制键盘鼠标的输入
import time
from kalmantrace import kalman
from filterpy.kalman import KalmanFilter
from utils import *
import matplotlib.pyplot as plt
#utils models都是yolo的 别管

x,y=(2560,1440) #屏幕大小
re_x,re_y=(2560,1440)
# 初始化图和坐标轴


init_x = np.array([[0], [0], [0], [0]])  # initial position and velocity
init_P = np.array([[1000, 0, 0, 0],
                   [0, 1000, 0, 0],
                   [0, 0, 1000, 0],
                   [0, 0, 0, 1000]])

trajectory = []
for i in range(6):
    trajectory.append((0,0))
det_3 = [0,10,10,0,0]

show_win=True #是否展示csgo-detect屏幕

device = "cuda:0" if torch.cuda.is_available() else 'cpu' #加速方式 yolo超参数之一
half = device != 'cpu' #yolo超参数之一
kk = 0
weights = r'E:\Yolo_project\yolov5-master\runs\train\exp6\weights\best.pt'
imgsz=640 #yolo超参数之一 预处理后图片大小
data='data/mydata.yaml' #yolo超参数之一 label
conf_thres=0.35 #yolo超参数之一
iou_thres=0.05
flag=0
mouse=pynput.mouse.Controller()
keyboard=pynput.keyboard.Controller()
stride=32
pt=True
names={0: 'ct', 1: 'ct_head', 2: 't', 3: 't_head'} #label

lock_mode=True #是否开锁
def on_move(x, y):
    pass

def on_click(x, y, button, pressed):
    global lock_mode
    if pressed and button == button.x2:
        lock_mode= not lock_mode
        print('lock_mode','on'if lock_mode else 'off')

def on_scroll(x, y, dx, dy):
    pass

def radar_plt(x_center,dist):
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N)) ** 2  # 0 to 15 point radii

    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()



# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 加载图像
image = Image.open('binary_map.jpg')



# 转化为灰度图像
#image = image.convert('L')
# 转化为numpy数组
image_data = np.array(image)
# 由于原图是二值化的，我们假设所有非0值（比如255）为可通行，同时假设0为不可通行
binary_map = np.where(image_data == 0, 0, 1)

end = (268,333)
count = 0


# class AStar:
#     def __init__(self, grid):
#         self.grid = grid
#         self.width = len(grid[0])
#         self.height = len(grid)
#
#     def heuristic(self, a, b):
#         return abs(a[0] - b[0]) + abs(a[1] - b[1])
#
#     def get_neighbors(self, node):
#         x, y = node
#         neighbors = []
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 if dx == 0 and dy == 0:
#                     continue
#                 new_x, new_y = x + dx, y + dy
#                 if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_y][new_x] == 1:
#                     neighbors.append((new_x, new_y))
#         return neighbors
#
#     def reconstruct_path(self, came_from, current):
#         path = [current]
#         while current in came_from:
#             current = came_from[current]
#             path.append(current)
#         return path
#
#     def find_path(self, start, end):
#         open_set = []
#         heapq.heappush(open_set, (0, start))  # (f, node)
#         came_from = {}
#         g_score = {start: 0}
#         f_score = {start: self.heuristic(start, end)}
#
#         while open_set:
#             _, current = heapq.heappop(open_set)  # We don't need the f value, it's only used for sorting
#             if current == end:
#                 return self.reconstruct_path(came_from, current)
#
#             for neighbor in self.get_neighbors(current):
#                 tentative_g_score = g_score[current] + 1
#                 if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
#                     came_from[neighbor] = current
#                     g_score[neighbor] = tentative_g_score
#                     f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
#                     heapq.heappush(open_set, (f_score[neighbor], neighbor))
#
#         return None  # No path found
#
#
class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.width = len(grid[0])
        self.height = len(grid)
        self.obstacle_value = 0  # 黑色部分的值，根据您的图像可能会有所不同

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_y][new_x] != self.obstacle_value:
                    neighbors.append((new_x, new_y))
        return neighbors

    def distance_to_obstacle(self,node):
        x, y = node
        mask = np.ones((7, 7))  # 以当前节点为中心的7x7掩码
        mask_x, mask_y = mask.shape
        min_distance = float('inf')

        for i in range(-3, 4):
            for j in range(-3, 4):
                if 0 <= x + i < self.width and 0 <= y + j < self.height:
                    if self.grid[y + j][x + i] == self.obstacle_value and mask[j + 3][i + 3] == 1:
                        # 如果发现黑色部分，计算到它的距离
                        distance = max(abs(i), abs(j))
                        min_distance = min(min_distance, distance)

        return min_distance

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path

    def find_path(self, start, end):
        open_set = []
        heapq.heappush(open_set, (0, start))  # (f, node)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)  # We don't need the f value, it's only used for sorting
            if current == end:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                if self.distance_to_obstacle(neighbor) >= 3:  # 检查距离限制
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

astar = AStar(binary_map)


# 函数来比较和更新数组
def compare_and_update(arr1, arr2, threshold=500):
    if arr1.shape != arr2.shape:
        print("Error: Arrays have different shapes.")
        return arr1

    diff = np.abs(arr1 - arr2)  # 计算两个数组对应位置的差值
    if np.any(diff > threshold):
        print("Error: Difference between arrays is greater than", threshold)
        return arr1
    else:
        return arr2  # 返回第二个数组作为更新后的结果

test = np.array([[0], [0]])
if_lock = 0
time.sleep(2)
path = []
count_path = 0
count_time = 0
while True:
    # 实时读取屏幕截图
    start_time_image=time.time()
    img0=grab_screen_mss(region=(0, 0, x, y))
    #img0=cv2.resize(img0,(re_x,re_y))
    # Load model
    if not flag:
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=half)
        #stride, names, pt = model.stride, model.names, model.pt
        #print(names)
        imgsz = check_img_size(imgsz, s=stride)
        flag=1
    '''
    if flag:
        device =  select_device(device)
        model = attempt_load(weights,device=device)
        imgsz = check_img_size(imgsz)
    '''
    #
    img = letterbox(img0, imgsz, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.


    if len(img.shape) == 3:
        img = img[None]  # img = img.unsqueeze(0)
    end_time_image=time.time()
    print("get image time",(end_time_image-start_time_image)*1000,'ms')
    torch.cuda.synchronize()
    start_time_pred=time.time()
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
    torch.cuda.synchronize()
    end_time_pred=time.time()
    print("pred time is",(end_time_pred-start_time_pred)*1000,"ms")
    #print(pred)
    aims=[]
    # # Process predictions
    flag_3_modified = False #为什么不丢到最上面去

    for i, det in enumerate(pred):
        flag_3_modified = False
        s=''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            for *xyxy, conf, cls in reversed(det):

                #bbox(tag,x_Ceter,y_Center,x_widt   h,y_width)
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
                #lock(aims, mouse, re_x, re_y)
                end_time_lock=time.time()
                #print("lock time",(end_time_lock-start_time_lock)*1000,"ms")
            for i,det in enumerate(aims):
                tag,x_center,y_center,width,height=det#supposed to be 'aim'
                x_center,y_center=re_x*float(x_center),re_y*float(y_center)
                width,height=re_x*float(width),re_y*float(height)
                top_left=(int(x_center-width/2), int(y_center-height/2))
                bottom_right=(int(x_center+width/2), int(y_center+height/2))
                color=(0,255,0) #RGB
                cv2.rectangle(img0,top_left,bottom_right,color,4)
                if tag == '2':
                    dist = 6063 / height - 4.97
                    #print(dist)
                if tag == '3':
                    lock(aims, mouse, re_x, re_y)
                    if_lock = 1
                max_trajectory_length = 10  # 设置你希望的最大轨迹点数量




                    # 清空trajectory列表以释放内存
                    #trajectory.clear()
                    #cv2.circle(img0, (int(init_x[0, 0]), int(init_x[2, 0])),3, color, 3)
                    #print((init_x[0, 0], init_x[2, 0]))

    if show_win:
        cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('csgo-detect', re_x // 4, re_y // 4)
        cv2.imshow('csgo-detect',img0)

        hwnd=win32gui.FindWindow(None,'csgo-detect')
        CVRECT=cv2.getWindowImageRect('csgo-detect')
        win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

    plt.clf()
    screen_img = grab_screen_mss(region=(34, 110, 314, 390))  # 定义屏幕区域 (30, 100, 310, 360)
    img1 = cv2.imread('image.png')  # 大地图

    # 调整图像大小
    resized_screen_img = cv2.resize(screen_img, (100, 100))  # 调整为所需大小
    img1 = cv2.resize(img1, (500, 500))

    # 找到图像中的关键点和描述符
    kp_screen, des_screen = sift.detectAndCompute(resized_screen_img, None)
    kp1, des1 = sift.detectAndCompute(img1, None)  # 大地图

    # 初始化暴力匹配器
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des_screen, k=2)

    # 应用比率测试
    good = []
    for m, n in matches:
        if m.distance < 0.62* n.distance:
            good.append(m)
    # 提取匹配点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_screen[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    direction =[]
    # 计算单应性矩阵
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5)  # 注意这里是小地图到大地图的转换
    if M is not None:
        theta_large_map = 90 - np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
    # 当前位置在小地图上的坐标，假设为小地图中心点
        current_position_small_map = np.array([[50, 50, 1]]).T  # 3x1坐标向量，因为小地图尺寸为100x100
    # 使用单应性矩阵将当前位置的坐标转换到大地图上
        current_position_large_map = np.dot(M, current_position_small_map)
    # 归一化处理，将齐次坐标转换为2D坐标
        current_position_large_map = (current_position_large_map / current_position_large_map[2])[:2]

    # 添加箭头以表示朝向
        arrow_length = 30
        center = (int(current_position_large_map[0]), int(current_position_large_map[1]))
        direction_offset = (int(arrow_length * np.cos(theta_large_map * np.pi / 180)),
                        int(-arrow_length * np.sin(theta_large_map * np.pi / 180)))
        end_point = (center[0] + direction_offset[0], center[1] + direction_offset[1])
        img_with_arrow = cv2.arrowedLine(img1, center, end_point, (0, 255, 0), 2)
        img_with_marker = cv2.circle(img_with_arrow, center, 3, (0, 0, 255), -1)
        #print("角度：", theta_large_map)
        if if_lock == 0:
            path_now = astar.find_path(center, end)
            if path_now is None:
                count_path += 1
                path20 = path[-5*(1+count_path):-5*(1+count_path)+1]
                path40 = path[-5*(2+count_path):-5*(2+count_path)+1]
            else:
                path20 = path_now[-10:-9]
                path40 = path_now[-10:-9]
                path = path_now
                count_path = 0

            fig, ax = plt.subplots()
            ax.imshow(binary_map, cmap='Greys', interpolation='nearest')

            # 绘制起点和终点
            ax.plot(center[0], center[1], 'o', color='green', markersize=10, label='Start')
            ax.plot(end[0], end[1], 'o', color='red', markersize=10, label='End')

            # 绘制路径
            if path:
                X, Y = zip(*path)
                ax.plot(X, Y, linestyle='-', color='blue', marker='o', markersize=4, label='Path')

            # 显示图例
            ax.legend()

            # 显示地图
            plt.show()
            test_current = current_position_large_map
            test_current1 = current_position_large_map#用于计算direction
            if(np.any(test==np.array([[0], [0]]))):
                test = test_current
            test=compare_and_update(test, test_current)
        # 计算差向量
            diff_vec = (path20[0][0] - center[0], - path20[0][1] + center[1])
        #print("差向量：", diff_vec)
        # 计算差向量与x轴正方向的夹角（角度）
            angle = math.atan2(diff_vec[1], diff_vec[0]) * 180 / math.pi

        # 将角度限制在 0 到 360 度之间
            angle = (angle + 360) % 360
        #print("角度差：", theta_large_map-angle)
        if if_lock == 0:
            #key = cv2.waitKey(400)  # 等待1秒，持续更新图像
            if (theta_large_map-angle) > 90 or (theta_large_map-angle) < -90:
                if theta_large_map-angle > 0:
                    target_x, target_y = 1280 + 7.744 * (20), 720
                else:
                    target_x, target_y = 1280 + 7.744 * (-20), 720
            else:

                target_x, target_y = 1280 + 7.94 * (theta_large_map-angle), 720
            mouse.position = (target_x, target_y)
            kb.press('w')
            time.sleep(0.5)
            kb.release('w')
        if_lock = 0
        #time.sleep(0.5)
    # 显示带有标记和箭头的图像
        print(current_position_large_map)
        cv2.imshow('Map with Current Position and Orientation', img_with_marker)
    else:
        print("No valid matches found. Homography matrix is None.")

    # # 目标位置
    # target_x, target_y = 1639, 720
    # #
    # # # 最终将鼠标移动到目标位置
    # mouse.position = (target_x, target_y)
    # 更新显示图像

    #if key == ord('q'):  # 如果按下 'q' 键，退出循环
      #  break

cv2.destroyAllWindows()  # 循环结束后关闭窗口


