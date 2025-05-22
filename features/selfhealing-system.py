import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

class SelfHealingSystem:
    def __init__(self):
        self.knowledge_base = self.load_knowledge_base()
        self.gnn_model = self.build_gnn_model()
        self.incident_history = []
        
    def build_gnn_model(self):
        node_features = tf.keras.Input(shape=(None, 10), name='node_features')
        edge_features = tf.keras.Input(shape=(None, 5), name='edge_features')
        adjacency = tf.keras.Input(shape=(None, None), name='adj_matrix', dtype='bool')
        
        x = GraphConv(64, activation='relu')([node_features, edge_features, adjacency])
        x = GraphPool()(x)
        x = GraphConv(32, activation='relu')(x)
        x = GraphGather()(x)
        
        outputs = Dense(3, activation='softmax')(x)
        
        return Model(inputs=[node_features, edge_features, adjacency],
                    outputs=outputs)
    
    def diagnose_issue(self, incident):
        service_graph = self.build_service_graph(incident)
        
        solution = self.gnn_model.predict(service_graph)
        
        if solution[0] > 0.7:
            self.execute_remediation(incident, solution)
        else:
            self.escalate_to_human(incident)
            
    def execute_remediation(self, incident, solution):
        if np.argmax(solution) == 0:
            self.restart_service(incident.service)
        elif np.argmax(solution) == 1:
            self.rollback_deployment(incident.deployment)
        elif np.argmax(solution) == 2:
            self.scale_resource(incident.resource)
            
        self.record_outcome(incident, solution, success=True)
    
    def learn_from_incidents(self):
        graph_data = self.prepare_training_data()
        self.gnn_model.fit(graph_data, epochs=10)