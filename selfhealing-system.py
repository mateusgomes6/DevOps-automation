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