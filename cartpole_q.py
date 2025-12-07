import gymnasium as gym


import numpy as np

NUM_DIZITIZED = 6


GAMMA = 0.99  # 時間割引率
ETA = 0.5  # 学習係数

class State:
    def __init__(self, num_states, num_actions):
        # 行動数を取得
        self.num_actions = num_actions

        # Qテーブルを作成　(分割数^状態数)×(行動数)
        self.q_table = np.random.uniform(low=-1, high=1, size=(NUM_DIZITIZED**num_states, num_actions))


    def bins(self, clip_min, clip_max, num):
        # 観測した状態デジタル変換する閾値を求める
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def analog2digitize(self, observation):
        #状態の離散化
        cart_pos, cart_v, pole_angle, pole_v = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-2.4, 2.4, NUM_DIZITIZED)),
            np.digitize(cart_v, bins=self.bins(-3.0, 3.0, NUM_DIZITIZED)),
            np.digitize(pole_angle, bins=self.bins(-0.5, 0.5, NUM_DIZITIZED)),
            np.digitize(pole_v, bins=self.bins(-2.0, 2.0, NUM_DIZITIZED))
        ]
        return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])

    def update_Q_table(self, observation, action, reward, observation_next):
        # 状態の離散化
        state = self.analog2digitize(observation)
        state_next = self.analog2digitize(observation_next)
        Max_Q_next = max(self.q_table[state_next][:])
        # Qテーブルを更新(Q学習)
        #print(self.q_table[state, action])
        self.q_table[state, action] = self.q_table[state, action]+ETA * (reward + GAMMA * Max_Q_next - self.q_table[state, action])
        #print(self.q_table[state, action])

    def decide_action(self, observation, episode):
        # ε-greedy法で行動を選択する
        state = self.analog2digitize(observation)
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            # 最も価値の高い行動を行う。
            action = np.argmax(self.q_table[state][:])
        else:
            # 適当に行動する。
            action = np.random.choice(self.num_actions)
        return action

class Agent:
    def __init__(self, num_states, num_actions):
        # 環境を生成
        self.state = State(num_states, num_actions)

    def update_Q_function(self, observation, action, reward, observation_next):
        # Qテーブルの更新
        #obsevationのprintをはさむ
        self.state.update_Q_table(observation, action, reward, observation_next)

    def get_action(self, observation, step):
        # 行動
        action = self.state.decide_action(observation, step)
        return action

import matplotlib.pyplot as plt
from matplotlib import animation


# 最大のステップ数
MAX_STEPS = 200
# 最大の試行回数
NUM_EPISODES = 10000



class Environment():
    def __init__(self, toy_env):
        # 環境を生成
        self.env = gym.make(toy_env)
        # 状態数を取得
        num_states = self.env.observation_space.shape[0]
        # 行動数を取得
        num_actions = self.env.action_space.n
        # Agentを生成
        self.agent = Agent(num_states, num_actions)
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")


    def run(self):
        complete_episodes = 0 # 成功数
        step_list = []
        is_episode_final = False  # 最後の試行
        frames = []  # 画像を保存する変数

        # 試行数分繰り返す
        for episode in range(NUM_EPISODES):
            #print("エピソード")
            observation = self.env.reset()  # 環境の初期化
            observation=observation[0]
            for step in range(MAX_STEPS):
                # 最後の試行のみ画像を保存する。
                if is_episode_final:
                    frames.append(self.env.render())
                    #print("hana")

                # 行動を求める
                action = self.agent.get_action(observation, episode)
                # 行動a_tの実行により、s_{t+1}, r_{t+1}を求める
                observation_next, _, done, _ ,_= self.env.step(action)
                #print(self.env.step(action))

                # 報酬を与える
                if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                    #print("done")
                    if step < 100:
                        reward = -1  # 失敗したので-1の報酬を与える
                        complete_episodes = 0  # 成功数をリセット
                        #print("ifのほう")
                        #print(reward)
                    else:
                        reward = 1  # 成功したので+1の報酬を与える
                        complete_episodes += 1  # 連続成功記録を更新
                        #print("elseのほう")
                        #print(reward)
                else:
                    reward = 0
                # Qテーブルを更新する
                self.agent.update_Q_function(observation, action, reward, observation_next)
                #print(reward)
                #print(self.agent.state.q_table)
                # 観測の更新
                
                observation = observation_next

                # 終了時の処理
                if done:
                    step_list.append(step+1)
                    #print(step_list)
                    #print('complete_episodes',complete_episodes)
                    break

            if is_episode_final:
            #if True:
                #print("night")
                es = np.arange(0, len(step_list))
                #print(es)
                #print()
                y=100*np.ones(len(step_list),dtype=int)
                plt.plot(es, step_list)
                plt.plot(es,y)
                plt.show()
                #plt.savefig("cartpole.png")
                #plt.figure()
                #patch = plt.imshow(frames[0])
                #plt.axis('off')


                def animate(i):
                    patch.set_data(frames[i])

                anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),interval=50)
                plt.show()
                
                '''anim.save('/content/drive/My Drive/lab activities/movie_cartpole_v0.mp4', writer="ffmpeg")'''

                break

            # 10連続成功したら最後の試行を行う
            if complete_episodes >= 80:
                print('80回連続成功')
                is_episode_final = True

TOY = "CartPole-v1"

def main():
    cartpole = Environment(TOY)
    cartpole.run()
    '''test=Agent(4,2)
    print(test.state.q_table)
    test.update_Q_function(np.array([-0.05012925, -0.3906783 ,  0.01841577,  0.6080843 ], dtype='float32'),0,0,np.array([-0.06185919, -0.00114511,  0.03700262,  0.0383729 ],dtype='float32'))
    print(test.state.q_table)'''

if __name__ == "__main__":
    main()


