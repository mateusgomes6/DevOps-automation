import tensorflow as tf
import prometheus_client

# Paths for submodules
server = prometheus_client.start_http_server
keras = tf.keras
layers = tf.keras.layers
models = tf.keras.models

# Paths for specific classes
LSTM = layers.LSTM 
Attention = layers.Attention

class AdvancedMonitoring:
    def __init__(self):
        self.model = self.build_advanced_model()
        self.metrics_server = server(8000)
        self.setup_custom_metrics()
