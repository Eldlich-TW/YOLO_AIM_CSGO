import time

import pynput

def lock(aims,mouse,x,y):
    dis_list=[]
    mouse_pos_x,mouse_pos_y=mouse.position
    for det in aims:
        _,x_center,y_center,_,_=det
        dis=(x*float(x_center) -mouse_pos_x)**2 + (y*float(y_center)-mouse_pos_y)**2
        dis_list.append(dis)
    det=aims[dis_list.index(min(dis_list))]

    tag,x_center,y_center,width,height=det

    x_center,y_center=x*float(x_center),y*float(y_center)
    width,height=x*float(width),y*float(height)
    if tag == "1" or tag == "3":
        mouse.position=(x_center,y_center)
        #time.sleep(0.01)
        #mouse.click(pynput.mouse.Button.left)
    #if tag == 0 or tag == 2:
        #mouse.position=(x_center,y_center-1/6*height)
