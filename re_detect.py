from grabscreen import *
import cv2
import win32gui
import win32con

x,y=(1920,1080) #屏幕大小
re_x,re_y=(1920,1080)

show_win=True

while True:
    img0=grab_screen(region=(0, 0, x, y))
    cv2.namedWindow('csgo-detect',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('csgo-detect',re_x//3,re_y//3)
    if show_win:
        cv2.imshow('csgo-detect',img0)
    hwnd=win32gui.FindWindow(None,'csgo-detect')
    CVRECT=cv2.getWindowImageRect('csgo-detect')
    win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    #win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE)

    if cv2.waitKey(1) & 0xFF==ord('q'): #input q to shutdown
        cv2.destroyAllWindows()
        break

