import tkinter as tt
import subprocess

def run_first_file():
    subprocess.run(['python', 'cs_model.py'])

def run_second_file():
    subprocess.run(['python', 'sift.py'])

def quit_app():
    root.destroy()

# 创建主窗口
root = tt.Tk()

# 创建按钮
button1 = tt.Button(root, text='执行第一个文件', command=run_first_file)
button1.pack()

button2 = tt.Button(root, text='执行第二个文件', command=run_second_file)
button2.pack()

button3 = tt.Button(root, text='退出', command=quit_app)
button3.pack()

# 启动主循环
root.mainloop()