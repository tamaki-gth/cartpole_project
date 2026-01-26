import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def save_animation(z_array,l,actions, filename="cartpole_qlearn.mp4"):
    x = z_array[:,0]
    x_dot=z_array[:,1]
    theta = z_array[:,2]
    theta_dot=z_array[:,3]
    

    xc = x + l*np.sin(theta)
    yc = l*np.cos(theta)

    fig, (ax_anim,ax_plot) = plt.subplots(2,1,figsize=(8,10))

    #アニメーションのグラフ
    ax_anim.set_xlim(np.min(x)-1, np.max(x)+1)
    ax_anim.set_ylim(-l, l)
    ax_anim.set_aspect('equal')

    cart_point, = ax_anim.plot([], [], "bo", markersize=8, label="cart")
    pole_point, = ax_anim.plot([], [], "ro", markersize=8, label="pole")

    #状態量のグラフ

    ax_left = ax_plot
    ax_right = ax_plot.twinx()
    
    ax_left.set_title("State variables over time")
    ax_left.set_xlabel("Frame")
    ax_left.set_ylabel("x, x_dot")
    ax_right.set_ylabel("theta, theta_dot")

    # 4本の線を準備（伸びていく）
    line_theta, = ax_right.plot([], [], label="theta", color="blue")
    line_x, = ax_left.plot([], [], label="x", color="green")
    line_theta_dot, = ax_right.plot([], [], label="theta_dot", color="red")
    line_x_dot, = ax_left.plot([], [], label="x_dot", color="purple")

    line_action, = ax_left.plot([], [], label="action", color="orange")


    lines = [line_x, line_x_dot, line_theta, line_theta_dot]
    labels = [l.get_label() for l in lines]
    ax_plot.legend(lines, labels, loc="upper right")


    

    def init():
        line_theta.set_data([], [])
        line_x.set_data([], [])
        line_theta_dot.set_data([], [])
        line_x_dot.set_data([], [])
        line_action.set_data([], []) 
        return line_theta, line_x, line_theta_dot, line_x_dot,line_action


    def update(i):

        #アニメーション
        ax_anim.clear()
        ax_anim.set_xlim(np.min(x)-1, np.max(x)+1)
        ax_anim.set_ylim(-l, l)
        ax_anim.set_aspect('equal')
        ax_anim.plot([x[i]], [0],"bo",markersize=8,label="cart")
        ax_anim.plot([xc[i]], [yc[i]],"ro",markersize=8,label="pole")
        
        #状態量
        line_theta.set_data(np.arange(i), theta[:i])
        line_x.set_data(np.arange(i), x[:i])
        line_theta_dot.set_data(np.arange(i), theta_dot[:i])
        line_x_dot.set_data(np.arange(i), x_dot[:i])

        line_action.set_data(np.arange(i), actions[:i])

        
        ax_plot.set_xlim(0, len(z_array))
        # y軸は自動調整
        ax_plot.relim()
        ax_plot.autoscale_view()

        ax_right.relim()
        ax_right.autoscale_view()


        return line_theta, line_x, line_theta_dot, line_x_dot,line_action


    ani = animation.FuncAnimation(
        fig, update, frames=len(z_array),
        init_func=init, interval=1, blit=False
    )
    ani.save("/home/tamaki/cartpole_mv.mp4",writer="ffmpeg",fps=100)
    
    plt.close()
