# coding=utf-8
import gym
from MyDQN import *

env_name = 'CartPole-v0'
model_name = "./model/cartpole_dqn.h5"
train_graph_name = "./graph/cartpole_train_dqn.png"
test_graph_name = "./graph/cartpole_test_dqn.png"

def train(graph_name, max_iter=20000, max_episodes=1000):
    env = gym.make(env_name)
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, batch_size=64, train_start=100)
    if os.path.exists(model_name):
        agent.model.load_weights(model_name)

    scores, losses = [], []


    for epoch in range(max_episodes):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        if np.mean(scores[-5:]) > 10000:
            break
        while score <= max_iter:
            if agent.render:
                agent.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            # 在CartPole_v0这个env中reward只有0/1两种,0只存在与done(游戏结束的情况)
            if done:
                reward = -100

            agent.add_memory(state, action, reward, done, next_state)
            agent.train_model()
            state = next_state

            if done or score >= max_iter:
                break
        if len(agent.losses_list) > 0:
            loss = np.array(agent.losses_list)
            print("(episode: {}; score: {}; memory length: {}; loss-mean: {})"
                      .format(epoch, score, len(agent.memory), loss.mean()))
            scores.append(score)
            losses.append(loss.mean())

        if epoch % 50 == 0:
            agent.model.save_weights(model_name)
    agent.model.save_weights(model_name)
    draw_plot(scores, losses, filename=graph_name)

def test(graph_name, max_iter=20000, max_episodes=100):
    env = gym.make(env_name)
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, batch_size=64, epsilon=0.0001)
    if os.path.exists(model_name):
        agent.model.load_weights(model_name)

    scores = []


    for epoch in range(max_episodes):
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while score <= max_iter:
            if agent.render:
                agent.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            # 在CartPole_v0这个env中reward只有0/1两种,0只存在与done(游戏结束的情况)


            state = next_state

            if done or score >= max_iter:
                print("(episode: {}; score: {};)"
                      .format(epoch, score))
                scores.append(score)
                break

    draw_score_plot(scores, filename=graph_name)


if __name__ == '__main__':
    train(train_graph_name, max_iter=20000, max_episodes=5000)
    test(test_graph_name, max_iter=20000, max_episodes=500)