import tensorflow as tf
import time
import numpy as np
import random


class BallAgent:
    def __init__(self, learning_rate=0.001, log_dir="./logs", epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_shape=(1,), activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')  # 2 ações: pular ou não pular
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

        # Epsilon-greedy parameters
        self.epsilon = epsilon  # Probabilidade inicial de exploração
        self.epsilon_min = epsilon_min  # Probabilidade mínima de exploração
        self.epsilon_decay = epsilon_decay  # Taxa de decaimento da exploração

        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Escolher uma ação aleatória para explorar
            return random.choice([0, 1])
        # Escolher a ação com maior valor Q (exploração)
        q_values = self.model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, step):
        q_values = self.model.predict(np.array([state]))
        next_q_values = self.model.predict(np.array([next_state]))
        q_values[0][action] = reward + 0.95 * np.max(next_q_values)

        # Log training data to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar('reward', reward, step=step)
            tf.summary.scalar('max_q_value', np.max(next_q_values), step=step)

        self.model.fit(np.array([state]), q_values, epochs=1, verbose=0)

        # Decair o epsilon para reduzir a exploração com o tempo
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path="ball_agent_model.h5"):
        self.model.save(path)

    def load_model(self, path="ball_agent_model.h5"):
        self.model = tf.keras.models.load_model(path)


class BallGame:
    def __init__(self, model_path="ball_agent_model.h5"):
        self.time_interval = 2  # O player deve pular a cada 2 segundos
        self.last_jump_time = time.time()
        self.score = 0
        self.agent = self.load_agent(model_path)

    def load_agent(self, model_path):
        # Carregar o modelo e passar a função de perda explicitamente
        return tf.keras.models.load_model(model_path, compile=False)

    def calculate_reward(self, time_since_last_jump):
        # Recompensa proporcional à proximidade com 2 segundos
        distance_from_ideal = abs(time_since_last_jump - self.time_interval)
        reward = 1 - distance_from_ideal  # Quanto mais perto de 2s, maior a recompensa
        return max(-1, reward)  # Limitar a penalidade mínima em -1

    def evaluate_jump(self):
        current_time = time.time()
        time_since_last_jump = current_time - self.last_jump_time

        # Definir estado (tempo desde o último salto)
        state = np.array([time_since_last_jump])

        # Obter valores Q do modelo e tomar uma decisão
        q_values = self.agent.predict(state)[0]
        action = np.argmax(q_values)

        print(f"Q-values: {q_values}, Ação Escolhida: {'Pular' if action == 1 else 'Não Pular'}")

        # Se ação for pular (1)
        if action == 1:
            reward = self.calculate_reward(time_since_last_jump)
            self.score += reward
            print(f"Rodada: Pulou após {time_since_last_jump:.2f} segundos. Recompensa: {reward:.2f}. Pontuação: {self.score:.2f}")
            # Resetar o tempo do último salto
            self.last_jump_time = current_time
        else:
            print("Rodada: A bola não pulou.")
            if time_since_last_jump > self.time_interval + 0.1:
                # Penalidade dobrada por não pular
                self.score -= 2
                print(f"Rodada: Não pulou a tempo! Penalidade de 2 pontos. Pontuação: {self.score:.2f}")
                # Resetar o tempo do último salto
                self.last_jump_time = current_time

    def run_game(self, duration=20):
        start_time = time.time()
        while time.time() - start_time < duration:
            current_time = time.time()
            time_since_last_jump = current_time - self.last_jump_time

            # Verificar se o tempo está próximo do intervalo de 2 segundos
            if time_since_last_jump >= 2.0:  # Verificar se chegou ao intervalo correto
                self.evaluate_jump()

            time.sleep(0.1)  # Checa a cada 0.1 segundos para suavidade


# Inicializando o agente e carregando o modelo treinado
game = BallGame("ball_agent_model.h5")
game.run_game(duration=20)  # Joga por 20 segundos
