import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

def rpd(action1, action2):

    point_list = np.array([[4,0], [10,1]])#得点[協力・協力の時,協力・裏切りの時],[裏切り・協力の時,裏切り]

    agent1_point = point_list[action1, action2]
    agent2_point = point_list[action2, action1]

    return agent1_point,agent2_point

class MM:
    def __init__(self):
        x = np.random.rand(1,2)
        self.model = self._softmax(x)
        self.alpha = 0.01
        
    def update(self, st):
        dist = [np.linalg.norm(m - st) for m in self.model]
        argmin = np.argmin(dist)
        self._update_param(argmin, st)
        return dist[argmin]
        
    def _update_param(self, argmin, st):
        self.model[argmin] = self.model[argmin]*(1-self.alpha)+st*self.alpha
        
    def _softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    
    def increase(self):
        pre_model = self.model
        self.model = np.random.rand(pre_model.shape[0]+1,pre_model.shape[1])
        for i,p in enumerate(pre_model):
            self.model[i]=p
        self.model[i+1]=p
        


class Agent_lean():
    def __init__(self,action_st):
        self.action_st = action_st
        #print(self.action_st)
        self.action_pre = 0
        print("学習エージェントの確率")
        if self.action_st == 0:
            l = [[0.7,0.3],[0.8,0.2],[0.9,0.1]]
            self.prob = np.array(random.choice(l))
            print(self.prob)
        else :
            l = [[0.1,0.9], [0.2,0.8],[0.3,0.7]]
            self.prob = np.array(random.choice(l))
            print(self.prob)
        
    def action(self):
        
        return np.random.choice(range(2), p = self.prob)
        
            #return np.random.choice(range(2), p = prob)
            #if self.action_pre == 0:
            #    self.action_pre = 1
            #    return [1]
            #else :
            #    self.action_pre = 1
            #    return [0]

class Agent():
    def __init__(self,action_list_num,belief,id):
        self.belief = belief
        self.id = id #各エージェントのID
        self.action_list_num = action_list_num
        self.agent_list =[]
        #self.expected_reward = 0 #期待報酬
        self.action_log = {ID : [0,0] for ID, i in enumerate(range(20))} #各エージェントの行動回数
        self.action_mean = {ID : [0] for ID, i in enumerate(range(20))} 
        self.agent_match = {ID : [0.5,0.5] for ID, i in enumerate(range(20))}
        self.sum_point = 0
        self._prob = []
        self.KL_p1 = []
        self.KL_p2 = []
        self.prob_total = []

        if belief == 0 :
            l = [[0.7,0.3],[0.8,0.2],[0.9,0.1]]
            self.prob = np.array(random.choice(l))
           
        else :
            l = [[0.1,0.9], [0.2,0.8],[0.3,0.7]]
            self.prob = np.array(random.choice(l))
        
        print(self.prob)
    def reset(self):
        self.sum_point = 0
        self.KL_p1 = []
        self.KL_p2 = []
        self.prob_total = []

    def action(self,self_action,other_action):

        action = np.random.choice(range(self.action_list_num), p = self.prob)    
        #if self.belief == 0:
            #action = 0
        #    action = np.random.choice(range(self.action_list_num), p = np.array([0.7,0.3]))
        #elif self.belief == 1 :
        #    action = np.random.choice(range(self.action_list_num), p = np.array([0.3,0.7]))
            #if self_action == other_action:
            #    action = np.random.choice(range(self.action_list_num), p = np.array([0.4,0.6]))
            #else :
            #    action = other_action 
       

        return action


    def prediction_action(self,mm_p1,mm_p2):
        action1 = np.random.choice(range(self.action_list_num), p = np.array([mm_p1, 1 - mm_p1]))                    
        action2 = np.random.choice(range(self.action_list_num), p = np.array([mm_p2, 1 - mm_p2]))     
       

        return action1,action2


    def agent_prediction(self,action,other_action):
        if self.belief == 0 :
            self.pre_action = 0
        elif self.belief == 1 :
            if other_action == action :
                self.pre_action = np.random.choice(range(self.action_list_num), p = np.array([0.3,0.7]))
            else :
                self.pre_action = action


    def count_action_log(self,action,otherID):
        other_action = np.identity(2)[action]
        self.action_log[otherID] += other_action
        self.log_mean(otherID)
        #print(self.action_log)


    def log_mean(self,otherid):
        sum_action = sum(self.action_log[otherid]) 
        self.action_mean[otherid] = self.action_log[otherid]/sum_action   
        #print(self.id,"vs",otherid,self.action_mean[otherid],) 
        #print("ゲーム数",sum_action) 


    def match_MM(self,otherid,mm_p1,mm_p2):
        prob = self.action_mean[otherid][0]
        if prob == 1:
            KL_p1 = (prob * (np.log(prob) - np.log(mm_p1)))  
            KL_p2 = (prob * (np.log(prob) - np.log(mm_p2))) 
        elif prob == 0:
            KL_p1 = (1 - prob) * (np.log((1 - prob)) - np.log((1 - mm_p1)))
            KL_p2 = (1 - prob) * (np.log((1 - prob)) - np.log((1 - mm_p2)))
        else :
            KL_p1 = (prob * (np.log(prob) - np.log(mm_p1))) + ((1 - prob) * (np.log((1 - prob)) - np.log((1 - mm_p1))))
            KL_p2 = (prob * (np.log(prob) - np.log(mm_p2))) + ((1 - prob) * (np.log((1 - prob)) - np.log((1 - mm_p2))))
        
        self.KL_p1.append(KL_p1)
        self.KL_p2.append(KL_p2)
        self.prob_total.append(prob)
        print("KLダイバージェンスの値",round(KL_p1,2),round(KL_p2,2))
        if KL_p1 < KL_p2:
            print(round(mm_p1,2),"の方が近い:",round(KL_p1,2))
        else :
            print(round(mm_p2,2),"の方が近い:",round(KL_p2,2))
        
        print("エージェントの行動確率",round(prob,2))

    def agent_predection(self,otherid,mm_p1,mm_p2,action,mm):
        #予測値を立てる
        action1,action2 = self.prediction_action(mm_p1,mm_p2)
        point1,a = rpd(action,action1)
 
        mm_point1 =  (point1 * self.agent_match[otherid][mm])
        
        #print("MM1の確率だと",action1,"MM2の確率だと",action2)
        return mm_point1

    def point_judge(self,mm_point1,mm_point2):
        if  self.sum_point < mm_point1 + mm_point2 :
            return False
        else :
            return True






def make_list(list,agent1,agent2):
    a = [agent1,agent2]
    list.append(a)
    return list


def list_delete(list,agent1,agent2):
    a = [agent1,agent2]
    list.remove(a)  
    return list


def plot(agents,game_list):
    G = nx.Graph()
    for agent in agents:
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
    nx.draw_networkx_nodes(G, pos, node_color = node_color,alpha=0.5)
    
    #ノード名を付加
    
    nx.draw_networkx_labels(G, pos,font_size=10)
    
    #X軸Y軸を表示しない設定
    plt.axis('off')
    
    #図を描画
    plt.show()


def get_MM():
    #a1 = [[1,0]]*10
    #a2 = [[0.4,0.6]]*10
    cooperation = [[0]]*10 #協力
    mutually = [[1]]*10 #交互
    action_list = np.vstack([cooperation,mutually])
    np.random.shuffle(action_list)
    agents = [Agent_lean(a) for a in action_list]
    model = MM()
    model.increase()
    dist=[]
    d=[]
    for a in range(100):
        for agent in agents:
            actions = [np.eye(2)[agent.action()] for i in range(25)]
            st = np.mean(actions,axis=0)
            dist.append(model.update(st))
        d.append(np.mean(dist))
        
        if a>10:
            if np.var(d[-10:]) <1e-8:
                model.increase()
    dist= np.array(d)   
    l = "dist"
    plt.plot(dist,label = l)
    plt.legend()
    plt.show()
    print(model.model)
    return model


def game_rpd(model):
    cooperation = [0]*5 #協力
    mutually = [1]*5 #交互
    mm_p1 = model.model[0][0]
    mm_p2 = model.model[1][0]
    print(mm_p1)
    print(mm_p2)
    beliefs = np.array(cooperation + mutually)# beliefsを決める
    agents =  [Agent(2,belief, id) for id, belief in enumerate(beliefs)]#agentの設定
    other_action = [random.randint(0,1),random.randint(0,1)] #agent1とagent2の一つ前のactionを格納
    comb_0 = []
    #comb_1 = []
    comb_0_after = []
    #comb_1_after = []
    for i in agents:
        for j in agents:
            if i.id == 0 and j.id != 0:
                make_list(comb_0,i,j)
                make_list(comb_0_after,i,j)
            #elif i.id == 10 and j.id != 10:
            #elif i.id == 10 and j.id != 10:
            #    make_list(comb_1,i,j)
            #    make_list(comb_1_after,i,j)

    plot(agents,comb_0)
    #plot(agents,comb_1)


    for agent1,agent2 in comb_0:
        print("agent1が",agent1.id)
        print(agent2.id)
        agent1.reset()
        sum_game = 0 #総ゲーム数 
        judge = True
        mm_point1 = 0
        mm_point2 = 0

        for i in range(25): #1エージェントの試合数
            if judge :
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
                #agent2.count_action_log(action1,agent1.id)
                agent1.match_MM(agent2.id,mm_p1,mm_p2) #どのMMに近いか,KLダイバージェンス
                #mm_point1 += agent1.agent_predection(agent2.id,mm_p1,mm_p2,action1,0) #値の予測
                #mm_point2 += agent1.agent_predection(agent2.id,mm_p2,mm_p1,action1,1)
                sum_game += 1
            else :
                comb_0_after = list_delete(comb_0_after,agent1,agent2)
                break
            
            #judge = agent1.point_judge(mm_point1,mm_point2)
        
        print(agent2.id,"とは",sum_game,"ゲームした")
        print(agent1.sum_point)

        prob = np.array(agent1.prob_total)
        l = agent2.belief,agent2.prob[0]
        plt.xlim(0, 25)
        plt.ylim(-0.5,1.3)
        plt.plot(prob,label = l)
        plt.legend()
        plt.show()

        plt.xlim(0, 25)
        plt.ylim(-0.5,2)
        l = "mm_p1",mm_p1
        KL = np.array(agent1.KL_p1)
        plt.plot(KL,label = l)
        #plt.show()
        
        l = "mm_p2",mm_p2
        KL = np.array(agent1.KL_p2)
        plt.plot(KL,label = l)
        plt.legend()
        plt.show()
        
    #plot(agents,comb_0_after)

    
    

    #plot(agents,comb_1_after)


def main():
    model = get_MM()
    game_rpd(model)


if __name__ == "__main__":
    main()