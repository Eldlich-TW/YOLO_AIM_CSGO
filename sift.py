import cv2
from grabscreen import grab_screen_mss
import numpy as np
import heapq
import matplotlib.pyplot as plt
import math
from PIL import Image
import time
import pynput  # 控制键盘鼠标的输入
import win32gui
import win32con


mouse = pynput.mouse.Controller()
from pynput import keyboard

kb = keyboard.Controller()

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 加载图像
image = Image.open('binary_map.jpg')

# 转化为灰度图像
# image = image.convert('L')
# 转化为numpy数组
image_data = np.array(image)
# 由于原图是二值化的，我们假设所有非0值（比如255）为可通行，同时假设0为不可通行
binary_map = np.where(image_data == 0, 0, 1)

end = (415, 110)
count = 0


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
                if 0 <= new_x < self.width and 0 <= new_y < self.height and self.grid[new_y][
                    new_x] != self.obstacle_value:
                    neighbors.append((new_x, new_y))
        return neighbors

    def distance_to_obstacle(self, node):
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

time.sleep(1.5)
path = []
count_path = 0
while True:
    # 实时读取屏幕截图
    plt.clf()
    screen_img = grab_screen_mss(region=(22, 80, 278, 336))  # 定义屏幕区域 (30, 100, 310, 360)
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
        if m.distance < 0.62 * n.distance:
            good.append(m)

    # 提取匹配点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_screen[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    direction = []
    # 计算单应性矩阵
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)  # 注意这里是小地图到大地图的转换
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
        # print("角度：", theta_large_map)

        path_now = astar.find_path(center, end)
        if path_now is None:
            count_path += 1
            path20 = path[-5 * (1 + count_path):-5 * (1 + count_path) + 1]
        else:
            path20 = path_now[-15:-14]
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
        # plt.show()

        # 在大地图上标记当前位置
        img_with_marker = cv2.circle(img_with_arrow, center, 3, (0, 0, 255), -1)
        test_current = current_position_large_map
        test_current1 = current_position_large_map  # 用于计算direction
        if (np.any(test == np.array([[0], [0]]))):
            test = test_current
        test = compare_and_update(test, test_current)
        # 计算差向量
        diff_vec = (path20[0][0] - center[0], - path20[0][1] + center[1])
        # print("差向量：", diff_vec)
        # 计算差向量与x轴正方向的夹角（角度）
        angle = math.atan2(diff_vec[1], diff_vec[0]) * 180 / math.pi

        # 将角度限制在 0 到 360 度之间
        angle = (angle + 360) % 360
        print("角度差：", theta_large_map - angle)
        if (theta_large_map - angle) > 90 or (theta_large_map - angle) < -90:
            if theta_large_map - angle > 0:
                target_x, target_y = 960 + 7.744 * (20), 540
            else:
                target_x, target_y = 960 - 7.744 * (20), 540
        else:

            target_x, target_y = 960 + 7.94 * (theta_large_map - angle), 540
        mouse.position = (target_x, target_y)
        kb.press('w')
        time.sleep(0.5)
        kb.release('w')
        # time.sleep(0.5)
        # 显示带有标记和箭头的图像
        print(current_position_large_map)
        # cv2.imshow('Map with Current Position and Orientation', img_with_marker)
        cv2.namedWindow('Map with Current Position and Orientation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Map with Current Position and Orientation', 1920 // 4, 1080 // 4)
        cv2.imshow('Map with Current Position and Orientation',img_with_marker)

        hwnd=win32gui.FindWindow(None,'Map with Current Position and Orientation')
        CVRECT=cv2.getWindowImageRect('Map with Current Position and Orientation')
        win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    else:
        print("No valid matches found. Homography matrix is None.")

    # # 目标位置
    # target_x, target_y = 1639, 720
    # #
    # # # 最终将鼠标移动到目标位置
    # mouse.position = (target_x, target_y)
    # 更新显示图像
    key = cv2.waitKey(200)  # 等待1秒，持续更新图像
    if key == ord('q'):  # 如果按下 'q' 键，退出循环
        break

cv2.destroyAllWindows()  # 循环结束后关闭窗口


