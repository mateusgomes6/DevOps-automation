from keras.models import Sequential
from keras.layers import Dense, Dropout

class CICDOrchestrator:
    def __init__(self):
        self.jenkins = Jenkins()
        self.gitlab = GitLab()
        self.model = self.build_deployment_model()
        self.performance_metrics = []
        
    def build_deployment_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(20,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model