import math
import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
L = 2.9 #[m]

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, a, delta):
        self.x = self.x + self.v * math.cos(self.yaw) * dt
        self.y = self.y + self.v * math.sin(self.yaw) * dt
        self.yaw = self.yaw + self.v / L * math.tan(delta)*dt
        self.v = self.v + a * dt

        print(self.yaw)


if __name__ == '__main__':

    print("Start Unicycle Model")
    T =100
    a = [1.0] * T
    delta = [math.radians(1.0)] * T

    state = State()

    x=[]
    y=[]
    yaw=[]
    v=[]
    time=[]
    t=0.0

    for (ai, di) in zip(a, delta):
        t = t+dt
        state.update(ai, di)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)

    plt.plot(x, y)
    plt.axis("equal")
    plt.grid(True)


    plt.show()


