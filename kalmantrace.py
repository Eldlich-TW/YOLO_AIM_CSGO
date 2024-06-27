import numpy as np

def kalman(prev_x, prev_P,aims,re_x,re_y):
    dt = 1.0
    A = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    R = 10 * np.eye(2)
    Q = np.eye(4)
    B = 0
    u = 0
    I = np.eye(4)

    x = prev_x
    P = prev_P
    list=[]
    # measurement update
    _,x_center,y_center,_,_=aims
    x_center,y_center=re_x*float(x_center),re_y*float(y_center)
    list=[x_center,y_center]
    Z = np.array([list])
    Y = Z.T - H.dot(x)
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H.T).dot(np.linalg.inv(S))
    x = x + K.dot(Y)
    P = (I - K.dot(H)).dot(P)

    # prediction
    x = A.dot(x) + B * u
    P = A.dot(P).dot(A.T) + Q

    return x, P