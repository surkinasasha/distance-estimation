class SimpleKalman:
    def __init__(self, initial_dist, q=0.05, r=1.0):
        self.x = initial_dist # Текущая оценка состояния
        self.P = 1.0          # Ошибка оценки
        self.Q = q            # Шум процесса
        self.R = r            # Шум измерения

    def update(self, measurement):
        # Этап предсказания
        self.P = self.P + self.Q
        
        # Этап коррекции 
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        
        return self.x