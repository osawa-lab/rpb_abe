from http.client import MULTI_STATUS
import random
import numpy as np
import itertools
import networkx as nx
import string
import matplotlib.pyplot as plt
from statistics import mean


gamma = 0.6
gamma_rand = 0.9

def rpd(action1, action2):

    point_list = np.array([[5,0], [10,1]])#得点[協力・協力の時,協力・裏切りの時],[裏切り・協力の時,裏切り]

    agent1_point = point_list[action1, action2]
    agent2_point = point_list[action2, action1]

    return agent1_point,agent2_point



class Agent():
    def __init__(self,action_list_num,belief,id):
        self.belief = belief
        self.id = id #各エージェントのID
        self.action_list_num = action_list_num
        self.agent_list =[]
        self.expected_reward = 0 #期待報酬
        self.action_log = {ID : [0,0] for ID, i in enumerate(range(10))} #各エージェントの行動回数
        self.action_mean = {ID : [0] for ID, i in enumerate(range(10))} 
        self.action_prob = [[0,0],[0,0]] #行動の確率[[協力][交互]]
        self.sum_point = 0
        self.likelifood = 2
        self.pre_action = 0
        self.random = False
        
    def reset(self):
        #self.sum_point = 0
        
        self.error = 0
        self.likelifood = 2
        self.pre_action = 0




    def action(self,self_action,other_action):
                        
        if self.belief == 0:
            action = 0
            #np.random.choice(range(self.action_list_num), p = np.array([0.9,0.1]))
        elif self.belief == 1 :
            if self_action == other_action:
                action = np.random.choice(range(self.action_list_num), p = np.array([0.5,0.5]))
            else :
                action = other_action 
       

        return action

    def judge_refusal(self,threshold):
        if self.likelifood <= threshold:
            return False
        else :
            return True
    
           

    def match_prediction(self,action):
        p = 2
        p_rand = 1
        if self.random :
            if self.pre_action == action:
                self.likelifood = self.likelifood * gamma_rand + p_rand *(1 - gamma_rand)
            else :
                self.likelifood = self.likelifood * gamma_rand
        
        else :
            if self.pre_action == action :
                self.likelifood = self.likelifood * gamma + p * (1 - gamma)
            else :
                self.likelifood = self.likelifood * gamma 
        

    def agent_prediction(self,action,other_action):
        if self.belief == 0 :
            self.pre_action = 0
        elif self.belief == 1 :
            if other_action == action :
                self.pre_action = np.random.choice(range(self.action_list_num), p = np.array([0.5,0.5]))
                self.random = True
            else :
                self.pre_action = action
                self.random = False
        
    def count_action_log(self,action,otherID):
        other_action = np.identity(2)[action]
        self.action_log[otherID] += other_action
        #print(self.action_log)


    def log_mean(self,otherid):
        sum_action = sum(self.action_log[otherid]) 
        self.action_mean[otherid] = self.action_log[otherid]/sum_action   
        print(self.action_mean)   




    






def list_delete(list,agent1,agent2):
    a = [agent1,agent2]
    list.remove(a)  
    return list

def make_list(list,agent1,agent2):
    a = [agent1,agent2]
    list.append(a)
    return list


def plot(agents,game_list):
    G = nx.Graph()
    
    for agent in agents:
        agent_id = agent.id
        if agent.belief == 0:
            G.add_node(agent,color = "red")
 
        elif agent.belief == 1:
            G.add_node(agent,color = "blue")
        else :
            G.add_node(agent,color = "green")


    G.add_edges_from(game_list)
    #図の作成。figsizeは図の大きさ
    plt.figure(figsize=(10, 8))
    
    #図のレイアウトを決める。kの値が小さい程図が密集する
    pos = nx.spring_layout(G, k=0.8)
    node_color = [node["color"] for node in G.nodes.values()]
    #ノードとエッジの描画
    # _color: 色の指定
    # alpha: 透明度の指定
    nx.draw_networkx_edges(G, pos, edge_color='y')
    nx.draw_networkx_nodes(G, pos, node_color= node_color,alpha=0.5)
    
    #ノード名を付加
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    #X軸Y軸を表示しない設定
    plt.axis('off')
    
    #図を描画
    plt.show()


for i in range(2):
    cooperation = [0]*5 #協力
    mutually = [1]*5
    #betrayal = [1]*4 #裏切り
    #Return = [2]*2 #しっぺ返し


    beliefs = np.array(cooperation + mutually)
    #np.random.shuffle(beliefs)
    agents =  [Agent(2,belief, id) for id, belief in enumerate(beliefs)]

    other_action = [random.randint(0,1),random.randint(0,1)] #agent1とagent2の一つ前のactionを格納

    threshold = 0.2 #閾値

    sum_game = 0 #総ゲーム数
    all = itertools.combinations(agents, 2)

    game_list =[]

    for agent1,agent2 in all:
        game_list = make_list(game_list,agent1,agent2)
        
        
    plot(agents,game_list)
 

    all = itertools.combinations(agents, 2)


    for agent1,agent2 in all:
        judge1 = True
        judge2 = True
        agent1.reset()
        agent2.reset()
        for i in range(25): #1エージェントの試合数
            agent1_point = 0 #このゲームで与えられるポイントの初期化
            agent2_point = 0 #このゲームで与えられるポイントの初期化
            agent1.agent_prediction(other_action[0],other_action[1]) #行動予測
            agent2.agent_prediction(other_action[1],other_action[0]) #行動予測
            if judge1 and judge2:
                action1= agent1.action(other_action[0],other_action[1]) #agent1のアクションを決定
                action2= agent2.action(other_action[1],other_action[0]) #agent2のアクションを決定
                #print(i+1,"回目")
                #print(agent1.belief,agent2.belief)
                #print(action1,action2)
                agent1_point,agent2_point = rpd(action1,action2)
                agent1.sum_point += agent1_point
                agent2.sum_point += agent2_point
                other_action = [action1,action2]
                agent1.count_action_log(action2,agent2.id)
                agent2.count_action_log(action1,agent1.id)
                agent1.match_prediction(other_action[1])#行動予測とあっているか
                agent2.match_prediction(other_action[0])
                sum_game += 1
            else :
                game_list = list_delete(game_list,agent1,agent2)
                break
            
            judge1 = agent1.judge_refusal(threshold)
            judge2 = agent2.judge_refusal(threshold)

        #print(agent1.sum_point,agent2.sum_point)
        agent1.log_mean(agent2.id)
        agent2.log_mean(agent1.id)
        

    

    for agent in agents:
        print("戦略",agent.belief,"ポイント",agent.sum_point)
    print("総ゲーム数",sum_game)
    plot(agents,game_list)

