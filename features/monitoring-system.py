import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LSTM, Attention
import prometheus_client
from prometheus_client import start_http_server
import json
from aiokafka import AIOKafkaConsumer

keras = tf.keras

class AdvancedMonitoring:
    def __init__(self):
        self.model = self.build_advanced_model()
        self.metrics_server = start_http_server(8000)
        self.setup_custom_metrics()

    def build_advanced_model(self):
        inputs = keras.Input(shape=(60, 15))
        x = LSTM(128, return_sequences=True)(inputs)
        x = Attention()([x, x])
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(3, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy', keras.metrics.AUC()])
        return model
    
    def setup_custom_metrics(self):
        self.cpu_prediction = prometheus_client.Gauge(
            'cpu_anomaly_score', 'Prediction score for CPU anomalies'
        )
        self.mem_prediction = prometheus_client.Gauge(
            'mem_anomaly_score', 'Prediction score for Memory anomalies'
        )
    
    def create_sequences(self, data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    async def stream_metrics(self):
        consumer = AIOKafkaConsumer(
            'server-metrics',
            bootstrap_servers='kafka:9092',
            value_deserializer=lambda v: json.loads(v.decode('utf-8')))
        
        await consumer.start()
        buffer = []
        
        try:
            async for msg in consumer:
                metrics = msg.value
                buffer.append(metrics)
                if len(buffer) >= 60:
                    sequence = self.create_sequences(buffer, 60)
                    prediction = self.model.predict(sequence)
                    self.analyze_prediction(prediction, buffer[-1])
                    buffer = buffer[-30:] 
        finally:
            await consumer.stop()

    def analyze_prediction(self, prediction, latest_metrics):
        self.cpu_prediction.set(prediction[0][2]) 
        self.mem_prediction.set(prediction[0][2])
        
        if prediction[0][2] > 0.95:
            self.trigger_incident(latest_metrics)