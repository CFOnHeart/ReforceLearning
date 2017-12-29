# coding=utf-8

from MyQLearning import *
import gym
import math
import numpy as np
from numpy import mean, ptp, var, std
import matplotlib.pyplot as plt

# init env variables
state_size = [4, 4, 4, 4, 10, 10]
observation_high = [1.01, 1.01, 1.01, 1.01, 12.6, 28.3]
observation_low = [-1.01, -1.01, -1.01, -1.01, -12.6, -28.3]

# init QLearning parameters
n_states = 1
for val in state_size:
    n_states *= val
actions = [0, 1, 2]
epsilon = 0.9
alpha = 0.3
gamma = 0.95

# experiment requirments
max_episodes = 100
max_iter = 2000

model_name = './model/acrobot_model'
env_name = "Acrobot-v1"

# 这里只做测试功能
def run(model_name, max_episodes = 1000, max_iter=2000, encode_func=encode_value_uniform):

    RL = QLearning(n_states, actions, epsilon=1.0)
    RL.read_model(model_name)

    # 创建环境
    env = gym.make(env_name)
    env = env.unwrapped   # 不做这个会有很多限制,比如这里就无法访问这个env中的x_threshold属性(还不知道为什么)
    # env.metadata['video.frames_per_second'] = 1
    # 用一个变量记录每一轮的timesteps
    iters = []

    for i_episode in range(max_episodes):
        observation = env.reset()
        s = encode_state(observation, observation_high, observation_low, 
            state_size, encode_value_func=encode_func)
        iter = 0
        while True:
            iter += 1
            # env.render()
            action = RL.choose_action(s)
            observation_, reward, done, info = env.step(action)

            if done:
                s_ = 'terminal'
            else:
                s_ = encode_state(observation_, observation_high, observation_low, 
                    state_size, encode_value_func=encode_func)

            # swap  
            s = s_

            # break while loop when end of this episode
            if done or iter >= max_iter:
                iters.append(iter)
                print("(episode:{}; iter: {})".format(i_episode, iter))
                break

    iters_ar = np.array(iters)
    print("mean: {}; var: {}; std: {}".format(np.mean(iters_ar), np.var(iters_ar), np.std(iters_ar)))

    return iters

def draw_plot(iter_uniform, iter_root, iter_square, graph_name='./graph/acrobot_v0_compare.png'):
    fig = plt.figure()
    ax =  fig.add_subplot(1, 1, 1)
    x1 = range(len(iter_uniform))
    x2 = range(len(iter_root))
    x3 = range(len(iter_square))

    # fig.xlabel('time')
    # fig.ylabel('score')
    ax.set_title('Compare three methods')

    ax.plot(x1,iter_uniform,marker="x",label="uniform")  
    ax.plot(x2,iter_root,marker="+",label="root")  
    ax.plot(x3,iter_square,marker="o",label="square") 
    ax.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.) 
    fig.savefig(graph_name)
    plt.show()


if __name__ == "__main__":
    iter_uniform = run('model/acrobot_v0_uniform_model', 
        max_episodes=1000, max_iter=2000, encode_func=encode_value_uniform);
    iter_root = run('model/acrobot_v0_root_model', 
        max_episodes=1000, max_iter=2000, encode_func=encode_value_root);
    iter_square = run('model/acrobot_v0_square_model', 
        max_episodes=1000, max_iter=2000, encode_func=encode_value_square);
    draw_plot(iter_uniform, iter_root, iter_square)

    