import numpy as np
import random
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, concatenate

class ContainerOrchestrator:
    def __init__(self):
        self.docker = docker.from_env()
        self.kubernetes = kubernetes.client.CoreV1Api()
        self.model = self.build_rl_model()
        self.memory = deque(maxlen=10000)
        
    def build_rl_model(self):
        state_input = Input(shape=(25,))
        x = Dense(128, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        
        value = Dense(1)(x)
        
        policy = Dense(5, activation='softmax')(x)
    
    def get_cluster_state(self):
        nodes = self.kubernetes.list_node()
        pods = self.kubernetes.list_pod_for_all_namespaces()
        
        state = []
        for node in nodes.items:
            node_metrics = self.get_node_metrics(node.metadata.name)
            state.extend([
                node_metrics['cpu'],
                node_metrics['memory'],
                node_metrics['pods'],
                node_metrics['network'],
                node_metrics['disk']
            ])
        return np.array(state)
    
    def take_action(self, action):
        if action == 1:
            self.scale_services()
        elif action == 2:
            self.optimize_resources()
        elif action == 3:
            self.rebalance_pods()
        elif action == 4:
            self.perform_maintenance()

    def train(self, episodes=1000):
        for e in range(episodes):
            state = self.get_cluster_state()
            value, policy = self.model.predict(state.reshape(1, -1))
            action = np.random.choice(5, p=policy[0])
            
            self.take_action(action)
            next_state = self.get_cluster_state()
            reward = self.calculate_reward(state, next_state)
            
            self.memory.append((state, action, reward, next_state))
            self.replay()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        
        values, policies = self.model.predict(states)
        next_values, _ = self.model.predict(next_states)
        
        targets = rewards + 0.95 * next_values
        self.model.fit(states, [targets, policies], verbose=0)