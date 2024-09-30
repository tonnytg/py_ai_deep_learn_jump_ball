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

        # Adicionar callback para o TensorBoard
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Escolher uma ação aleatória para explorar
            return random.choice([0, 1])
        # Escolher a ação com maior valor Q (exploração)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, step):
        q_values = self.model.predict(np.array([state]), verbose=0)
        next_q_values = self.model.predict(np.array([next_state]), verbose=0)
        target_q_value = reward + 0.95 * np.max(next_q_values)
        q_values[0][action] = target_q_value

        # Log training data to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar('reward', reward, step=step)
            tf.summary.scalar('epsilon', self.epsilon, step=step)
            tf.summary.scalar('max_q_value', np.max(next_q_values), step=step)
            tf.summary.scalar('target_q_value', target_q_value, step=step)

        # Treinar o modelo no estado atual
        history = self.model.fit(np.array([state]), q_values, epochs=1, verbose=0, callbacks=[self.tensorboard_callback])

        # Decair o epsilon para reduzir a exploração com o tempo
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path="ball_agent_model.h5"):
        self.model.save(path)

    def load_model(self, path="ball_agent_model.h5"):
        self.model = tf.keras.models.load_model(path)


class BallGame:
    def __init__(self, agent, model_path="ball_agent_model.h5"):
        self.time_interval = 2  # O player deve pular a cada 2 segundos
        self.last_jump_time = time.time()
        self.score = 0
        self.agent = agent
        self.step = 0

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

        # Obter ação do agente
        action = self.agent.act(state)

        # Se ação for pular (1)
        if action == 1:
            reward = self.calculate_reward(time_since_last_jump)
            next_state = np.array([0])  # Após o pulo, o tempo é resetado
            self.agent.train(state, action, reward, next_state, self.step)
            self.score += reward
            print(f"Pulou após {time_since_last_jump:.2f} segundos. Recompensa: {reward:.2f}. Pontuação: {self.score:.2f}")
            self.last_jump_time = current_time
        else:
            print("Rodada: A bola não pulou.")
            if time_since_last_jump > self.time_interval + 0.1:
                # Penalidade dobrada por não pular
                reward = -2
                next_state = np.array([0])
                self.agent.train(state, action, reward, next_state, self.step)
                self.score += reward
                print(f"Não pulou a tempo! Penalidade de {reward} pontos. Pontuação: {self.score:.2f}")
                self.last_jump_time = current_time

        self.step += 1

    def run_game(self, duration=20):
        start_time = time.time()
        while time.time() - start_time < duration:
            current_time = time.time()
            time_since_last_jump = current_time - self.last_jump_time

            # Verificar se o tempo está próximo do intervalo de 2 segundos
            if time_since_last_jump >= 2.0:
                self.evaluate_jump()

            time.sleep(0.1)  # Checa a cada 0.1 segundos para suavidade


# Inicializando o agente e o treinamento
agent = BallAgent()

# Rodar o jogo com o agente treinado
game = BallGame(agent)
game.run_game(duration=20)  # Joga por 20 segundos
