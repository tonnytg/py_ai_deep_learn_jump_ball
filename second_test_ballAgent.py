import tensorflow as tf
import numpy as np
import time
import os


class BallAgent:
    def __init__(self, learning_rate=0.001, log_dir="./logs"):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_shape=(1,), activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')  # 2 ações: pular ou não pular
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

        # TensorBoard log
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def act(self, state):
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


# Exemplo de uso
agent = BallAgent()

# Simulação de treinamento (só um exemplo simplificado)
for episode in range(100):
    state = np.random.random()  # Estado aleatório (pode ser o tempo desde o último salto)
    action = agent.act(state)

    # Definir uma recompensa baseada na ação e no estado
    if action == 1:  # Pular
        reward = 1 if 1.9 <= state <= 2.1 else -1  # Pular no tempo correto dá 1 ponto
    else:
        reward = -1  # Não pular também pode ser uma penalidade

    next_state = np.random.random()  # Próximo estado (simulação)

    agent.train(state, action, reward, next_state, episode)

    print(f"Episode {episode + 1} - Action: {action}, Reward: {reward}")

# Para visualizar o treinamento:
# No terminal, execute: tensorboard --logdir=./logs
