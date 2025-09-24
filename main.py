import pygame
import numpy as np
import pickle
import os
from collections import defaultdict
import random
import math

# Инициализация Pygame
pygame.init()

# Константы
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 60
FIREBALL_SIZE = 12
PLAYER_SPEED = 5
FIREBALL_SPEED = 8
MAX_MANA = 100
MANA_REGEN = 0.8
FIREBALL_COST = 25
EXPLOSION_SIZE = 50

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
DARK_RED = (139, 0, 0)


class Fireball:
    def __init__(self, x, y, target_x, target_y, owner):
        self.x = x
        self.y = y
        self.owner = owner

        # Вычисляем направление к цели
        dx = target_x - x
        dy = target_y - y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > 0:
            self.vel_x = (dx / distance) * FIREBALL_SPEED
            self.vel_y = (dy / distance) * FIREBALL_SPEED
        else:
            self.vel_x = FIREBALL_SPEED
            self.vel_y = 0

        self.lifetime = 150  # Время жизни снаряда

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.lifetime -= 1

        return self.lifetime > 0

    def draw(self, screen):
        # Разные цвета снарядов для игрока и ИИ
        if self.owner == 'player':
            colors = [BLUE, (0, 150, 255), (150, 200, 255)]  # Синие тона
        else:  # ИИ
            colors = [RED, ORANGE, YELLOW]  # Красно-оранжевые тона

        sizes = [FIREBALL_SIZE, FIREBALL_SIZE - 3, FIREBALL_SIZE - 6]

        for i, (color, size) in enumerate(zip(colors, sizes)):
            if size > 0:
                pygame.draw.circle(screen, color, (int(self.x), int(self.y)), size)


class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 0
        self.max_size = EXPLOSION_SIZE
        self.lifetime = 20

    def update(self):
        if self.size < self.max_size:
            self.size += 3
        self.lifetime -= 1
        return self.lifetime > 0

    def draw(self, screen):
        alpha = max(0, self.lifetime * 12)
        colors = [(255, 0, 0, alpha), (255, 165, 0, alpha), (255, 255, 0, alpha)]

        for i, color in enumerate(colors):
            size = max(0, self.size - i * 5)
            if size > 0:
                s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, color, (size, size), size)
                screen.blit(s, (self.x - size, self.y - size))


class QLearningAI:
    def __init__(self, learning_rate=0.25, discount_factor=0.9, epsilon=0.9):
        self.q_table = defaultdict(lambda: [0.0] * 9)  # 9 действий (добавили защитную стрельбу)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995  # Медленнее снижение для больше исследований
        self.min_epsilon = 0.1  # Выше минимум для постоянного обучения

        self.last_state = None
        self.last_action = None
        self.last_reward = 0

        # Статистика для анализа
        self.shots_fired = 0
        self.hits_scored = 0
        self.deaths = 0

        # Система бесконечных бонусов за убийства игрока
        self.kill_bonuses = 0
        self.speed_multiplier = 1.0
        self.accuracy_bonus = 0.0
        self.reaction_bonus = 0.0
        self.defensive_bonus = 0.0  # Новый бонус для защитной стрельбы

        # Непрерывное имитационное обучение
        self.player_behavior_data = {
            'movement_patterns': [],
            'dodge_timings': [],
            'shooting_patterns': [],
            'preferred_distances': [],
            'reaction_times': [],
            'evasion_directions': []
        }
        self.imitation_learning_rate = 0.15  # Увеличили скорость имитации
        self.continuous_learning = True

    def apply_kill_bonus(self):
        """Применяет БЕСКОНЕЧНЫЕ бонусы за убийство игрока"""
        self.kill_bonuses += 1

        # Бесконечное увеличение скорости (без ограничений!)
        self.speed_multiplier += 0.15  # +15% за каждое убийство

        # Бесконечное увеличение точности стрельбы
        self.accuracy_bonus += 0.08  # +8% за каждое убийство

        # Улучшение времени реакции (бесконечное ускорение)
        self.reaction_bonus += 0.12  # +12% за каждое убийство

        # Новый бонус защитной стрельбы
        self.defensive_bonus += 0.1  # +10% шанс защитной стрельбы

        # МЕГА-награда за убийство + экспоненциальный бонус к обучению
        base_reward = 300
        kill_multiplier = 1 + (self.kill_bonuses * 0.5)  # Увеличивающиеся награды
        total_reward = int(base_reward * kill_multiplier)

        if self.last_state is not None:
            # Супер-усиленное обучение после успеха
            current_q = self.q_table[self.last_state][self.last_action]
            learning_boost = 0.3 + (self.kill_bonuses * 0.05)  # Увеличиваем скорость обучения
            self.q_table[self.last_state][self.last_action] = current_q + (total_reward * learning_boost)

        # Усиливаем имитационное обучение после каждого убийства
        self.imitation_learning_rate = min(0.5, self.imitation_learning_rate + 0.02)

        print(f"🔥 ИИ ЭВОЛЮЦИОНИРОВАЛ! Убийств: {self.kill_bonuses}")
        print(f"   💨 Скорость: {self.speed_multiplier:.2f}x")
        print(f"   🎯 Точность: +{self.accuracy_bonus:.2f}")
        print(f"   ⚡ Реакция: +{self.reaction_bonus:.2f}")
        print(f"   🛡️ Защита: +{self.defensive_bonus:.2f}")

        # Постоянное обучение - никогда не останавливаемся
        self.epsilon = max(self.min_epsilon, self.epsilon + 0.1)  # Увеличиваем исследование после успеха!

    def observe_player_behavior(self, player_pos, player_last_pos, player_action, danger_nearby):
        """Наблюдает за поведением игрока для имитационного обучения"""

        # Записываем паттерны движения
        if player_last_pos is not None:
            movement = (player_pos[0] - player_last_pos[0], player_pos[1] - player_last_pos[1])
            self.player_behavior_data['movement_patterns'].append(movement)

            # Ограничиваем размер истории
            if len(self.player_behavior_data['movement_patterns']) > 100:
                self.player_behavior_data['movement_patterns'].pop(0)

            # Записываем тайминг уклонений (только если есть предыдущая позиция)
            if danger_nearby and (
                    abs(player_pos[0] - player_last_pos[0]) > 3 or abs(player_pos[1] - player_last_pos[1]) > 3):
                self.player_behavior_data['dodge_timings'].append(danger_nearby)
                if len(self.player_behavior_data['dodge_timings']) > 50:
                    self.player_behavior_data['dodge_timings'].pop(0)

        # Анализируем и улучшаем Q-таблицу на основе успешных действий игрока
        if len(self.player_behavior_data['movement_patterns']) > 10:
            self.apply_imitation_learning()

    def apply_imitation_learning(self):
        """Применяет имитационное обучение на основе наблюдений за игроком"""

        # Анализируем успешные паттерны движения игрока
        recent_moves = self.player_behavior_data['movement_patterns'][-20:]

        # Находим наиболее часто используемые движения
        move_frequency = defaultdict(int)
        for move in recent_moves:
            # Квантуем движения для анализа
            quantized_move = (int(move[0] / 5) * 5, int(move[1] / 5) * 5)
            move_frequency[quantized_move] += 1

    def get_state(self, player_pos, ai_pos, fireballs, ai_mana, player_mana):
        # Более детальное состояние для лучшего обучения
        px, py = player_pos
        ax, ay = ai_pos

        # Позиции игроков (дискретизированные)
        player_zone_x = min(4, max(0, int(px / (SCREEN_WIDTH / 5))))
        player_zone_y = min(4, max(0, int(py / (SCREEN_HEIGHT / 5))))
        ai_zone_x = min(4, max(0, int(ax / (SCREEN_WIDTH / 5))))
        ai_zone_y = min(4, max(0, int(ay / (SCREEN_HEIGHT / 5))))

        # Ближайший опасный снаряд
        closest_danger = None
        min_distance = float('inf')

        for fireball in fireballs:
            if fireball.owner != 'ai':  # Снаряды игрока опасны для ИИ
                distance = math.sqrt((fireball.x - ax) ** 2 + (fireball.y - ay) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_danger = fireball

        if closest_danger:
            danger_zone_x = min(4, max(0, int(closest_danger.x / (SCREEN_WIDTH / 5))))
            danger_zone_y = min(4, max(0, int(closest_danger.y / (SCREEN_HEIGHT / 5))))
            danger_close = 1 if min_distance < 100 else 0

            # Направление движения опасного снаряда
            danger_vel_x = 1 if closest_danger.vel_x > 0 else 0
            danger_vel_y = 1 if closest_danger.vel_y > 0 else 0
        else:
            danger_zone_x = danger_zone_y = danger_close = danger_vel_x = danger_vel_y = 0

        # Состояние маны
        mana_state = min(3, int(ai_mana / 35))  # 0-3
        can_shoot = 1 if ai_mana >= FIREBALL_COST else 0

        return (player_zone_x, player_zone_y, ai_zone_x, ai_zone_y,
                danger_zone_x, danger_zone_y, danger_close,
                danger_vel_x, danger_vel_y, mana_state, can_shoot)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 7)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state]) if next_state else 0
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self, filename="fireball_ai_model.pkl"):
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'shots_fired': self.shots_fired,
            'hits_scored': self.hits_scored,
            'deaths': self.deaths,
            'kill_bonuses': self.kill_bonuses,
            'speed_multiplier': self.speed_multiplier,
            'accuracy_bonus': self.accuracy_bonus,
            'reaction_bonus': self.reaction_bonus,
            'defensive_bonus': self.defensive_bonus,
            'player_behavior_data': self.player_behavior_data,
            'imitation_learning_rate': self.imitation_learning_rate
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self, filename="fireball_ai_model.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = defaultdict(lambda: [0.0] * 9, data.get('q_table', {}))
                self.epsilon = data.get('epsilon', self.epsilon)
                self.shots_fired = data.get('shots_fired', 0)
                self.hits_scored = data.get('hits_scored', 0)
                self.deaths = data.get('deaths', 0)
                self.kill_bonuses = data.get('kill_bonuses', 0)
                self.speed_multiplier = data.get('speed_multiplier', 1.0)
                self.accuracy_bonus = data.get('accuracy_bonus', 0.0)
                self.reaction_bonus = data.get('reaction_bonus', 0.0)
                self.defensive_bonus = data.get('defensive_bonus', 0.0)
                self.imitation_learning_rate = data.get('imitation_learning_rate', 0.15)
                self.player_behavior_data = data.get('player_behavior_data', {
                    'movement_patterns': [],
                    'dodge_timings': [],
                    'shooting_patterns': [],
                    'preferred_distances': [],
                    'reaction_times': [],
                    'evasion_directions': []
                })

                # Убеждаемся что все ключи присутствуют
                required_keys = ['movement_patterns', 'dodge_timings', 'shooting_patterns',
                                 'preferred_distances', 'reaction_times', 'evasion_directions']
                for key in required_keys:
                    if key not in self.player_behavior_data:
                        self.player_behavior_data[key] = []
            print(f"🤖 Модель ИИ загружена!")
            print(f"   🏆 Убийств: {self.kill_bonuses}")
            print(f"   💨 Скорость: {self.speed_multiplier:.2f}x")
            print(f"   🎯 Точность: +{self.accuracy_bonus:.2f}")
            print(f"   🛡️ Защита: +{self.defensive_bonus:.2f}")


class Player:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.mana = MAX_MANA
        self.deaths = 0
        self.last_shot_time = 0
        self.invulnerable_time = 0

    def update(self):
        # Регенерация маны
        self.mana = min(MAX_MANA, self.mana + MANA_REGEN)

        # Уменьшение времени неуязвимости
        if self.invulnerable_time > 0:
            self.invulnerable_time -= 1

    def can_shoot(self):
        return self.mana >= FIREBALL_COST

    def shoot(self, target_x, target_y):
        if self.can_shoot():
            self.mana -= FIREBALL_COST
            return Fireball(self.x + PLAYER_WIDTH // 2, self.y + PLAYER_HEIGHT // 2,
                            target_x, target_y, 'player' if self.color == BLUE else 'ai')
        return None

    def take_damage(self):
        if self.invulnerable_time <= 0:
            self.deaths += 1
            self.invulnerable_time = 90  # 1.5 секунды неуязвимости
            return True
        return False

    def draw(self, screen):
        # Эффект мигания при неуязвимости
        if self.invulnerable_time > 0 and self.invulnerable_time % 10 < 5:
            color = (self.color[0] // 2, self.color[1] // 2, self.color[2] // 2)
        else:
            color = self.color

        pygame.draw.rect(screen, color, (self.x, self.y, PLAYER_WIDTH, PLAYER_HEIGHT))

        # Рисуем полоску маны
        mana_width = int((self.mana / MAX_MANA) * PLAYER_WIDTH)
        mana_color = PURPLE if self.mana >= FIREBALL_COST else RED
        pygame.draw.rect(screen, mana_color, (self.x, self.y - 10, mana_width, 5))


class FireballDuel:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Дуэль огненных шаров - Самообучающийся ИИ")
        self.clock = pygame.time.Clock()

        self.player = Player(50, SCREEN_HEIGHT // 2, BLUE)
        self.ai_player = Player(SCREEN_WIDTH - 50 - PLAYER_WIDTH, SCREEN_HEIGHT // 2, RED)

        self.fireballs = []
        self.explosions = []

        self.ai = QLearningAI()
        self.ai.load_model()

        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 20)

        # Для управления ИИ и отслеживания игрока
        self.ai_last_action_time = 0
        self.ai_action_delay = 8  # Задержка между действиями ИИ

        # Отслеживание поведения игрока
        self.player_last_pos = None
        self.player_last_shot_time = 0
        self.frame_counter = 0

    def handle_input(self):
        keys = pygame.key.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        mouse_pressed = pygame.mouse.get_pressed()[0]

        # Движение игрока
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.player.y = max(0, self.player.y - PLAYER_SPEED)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player.y = min(SCREEN_HEIGHT - PLAYER_HEIGHT, self.player.y + PLAYER_SPEED)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.player.x = max(0, self.player.x - PLAYER_SPEED)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.player.x = min(SCREEN_WIDTH - PLAYER_WIDTH, self.player.x + PLAYER_SPEED)

        # Стрельба игрока
        if mouse_pressed:
            fireball = self.player.shoot(mouse_pos[0], mouse_pos[1])
            if fireball:
                self.fireballs.append(fireball)
                self.player_last_shot_time = self.frame_counter
                self._player_shot_this_frame = True  # Отмечаем что игрок стрелял

                # ИИ наблюдает за стрельбой игрока
                shoot_distance = math.sqrt((mouse_pos[0] - self.player.x) ** 2 + (mouse_pos[1] - self.player.y) ** 2)
                self.ai.player_behavior_data['shooting_patterns'].append({
                    'distance': shoot_distance,
                    'target_pos': mouse_pos,
                    'player_pos': (self.player.x, self.player.y)
                })

                if len(self.ai.player_behavior_data['shooting_patterns']) > 50:
                    self.ai.player_behavior_data['shooting_patterns'].pop(0)

    def update_ai(self):
        # Применяем бонус скорости к задержке действий
        adjusted_delay = max(2, int(self.ai_action_delay * (1 - self.ai.reaction_bonus)))

        if self.ai_last_action_time <= 0:
            # Получаем состояние
            state = self.ai.get_state(
                (self.player.x + PLAYER_WIDTH // 2, self.player.y + PLAYER_HEIGHT // 2),
                (self.ai_player.x + PLAYER_WIDTH // 2, self.ai_player.y + PLAYER_HEIGHT // 2),
                self.fireballs,
                self.ai_player.mana,
                self.player.mana
            )

            # Выбираем действие
            action = self.ai.choose_action(state)

            # Применяем действие
            self.apply_ai_action(action)

            # Обучение
            if self.ai.last_state is not None:
                reward = self.calculate_ai_reward()
                self.ai.update_q_table(self.ai.last_state, self.ai.last_action, reward, state)

            # НЕПРЕРЫВНОЕ наблюдение за игроком - каждый кадр!
            current_player_pos = (self.player.x + PLAYER_WIDTH // 2, self.player.y + PLAYER_HEIGHT // 2)

            # Проверяем, есть ли опасность рядом с игроком
            danger_nearby = 0
            for fireball in self.fireballs:
                if fireball.owner == 'ai':
                    distance = math.sqrt(
                        (fireball.x - current_player_pos[0]) ** 2 + (fireball.y - current_player_pos[1]) ** 2)
                    if distance < 120:  # Увеличили радиус детекции
                        danger_nearby = distance
                        break

            # Проверяем, стрелял ли игрок на этом кадре
            player_shot = False
            if hasattr(self, '_player_shot_this_frame'):
                player_shot = self._player_shot_this_frame
                delattr(self, '_player_shot_this_frame')

            # Постоянное наблюдение
            self.ai.observe_player_behavior(current_player_pos, self.player_last_pos, danger_nearby, player_shot)
            self.player_last_pos = current_player_pos

            self.ai.last_state = state
            self.ai.last_action = action
            self.ai_last_action_time = adjusted_delay
        else:
            self.ai_last_action_time -= 1

    def apply_ai_action(self, action):
        # Применяем бесконечный бонус скорости к движению
        speed = PLAYER_SPEED * self.ai.speed_multiplier

        # 0: вверх, 1: вниз, 2: влево, 3: вправо,
        # 4: стрелять в игрока, 5: стрелять с упреждением, 6: уклонение, 7: ничего не делать
        # 8: ЗАЩИТНАЯ СТРЕЛЬБА (новое действие!)

        if action == 0:  # Вверх
            self.ai_player.y = max(0, self.ai_player.y - speed)
        elif action == 1:  # Вниз
            self.ai_player.y = min(SCREEN_HEIGHT - PLAYER_HEIGHT, self.ai_player.y + speed)
        elif action == 2:  # Влево
            self.ai_player.x = max(0, self.ai_player.x - speed)
        elif action == 3:  # Вправо
            self.ai_player.x = min(SCREEN_WIDTH - PLAYER_WIDTH, self.ai_player.x + speed)
        elif action == 4:  # Стрелять прямо в игрока
            # Применяем бонус точности
            target_x = self.player.x + PLAYER_WIDTH // 2
            target_y = self.player.y + PLAYER_HEIGHT // 2

            # Добавляем небольшую корректировку на основе бонуса точности
            if self.ai.accuracy_bonus > 0:
                # Анализируем последние движения игрока для предсказания
                if len(self.ai.player_behavior_data['movement_patterns']) > 5:
                    recent_moves = self.ai.player_behavior_data['movement_patterns'][-5:]
                    avg_move_x = sum(move[0] for move in recent_moves) / len(recent_moves)
                    avg_move_y = sum(move[1] for move in recent_moves) / len(recent_moves)

                    # Корректируем цель на основе паттерна движения
                    prediction_frames = int(10 + self.ai.accuracy_bonus * 20)
                    target_x += avg_move_x * prediction_frames
                    target_y += avg_move_y * prediction_frames

            fireball = self.ai_player.shoot(target_x, target_y)
            if fireball:
                self.fireballs.append(fireball)
                self.ai.shots_fired += 1
        elif action == 5:  # Стрелять с упреждением
            # Улучшенное упреждение с учетом обученных паттернов
            predict_frames = int(30 + self.ai.accuracy_bonus * 40)

            future_x = self.player.x + PLAYER_WIDTH // 2
            future_y = self.player.y + PLAYER_HEIGHT // 2

            # Используем данные о поведении игрока для лучшего предсказания
            if len(self.ai.player_behavior_data['movement_patterns']) > 3:
                recent_moves = self.ai.player_behavior_data['movement_patterns'][-3:]
                avg_vel_x = sum(move[0] for move in recent_moves) / len(recent_moves)
                avg_vel_y = sum(move[1] for move in recent_moves) / len(recent_moves)

                future_x += avg_vel_x * predict_frames
                future_y += avg_vel_y * predict_frames

            fireball = self.ai_player.shoot(future_x, future_y)
            if fireball:
                self.fireballs.append(fireball)
                self.ai.shots_fired += 1
        elif action == 6:  # Экстренное уклонение
            # Находим ближайшую опасность и уклоняемся
            closest_fireball = None
            min_distance = float('inf')

            for fireball in self.fireballs:
                if fireball.owner != 'ai':
                    distance = math.sqrt((fireball.x - self.ai_player.x) ** 2 + (fireball.y - self.ai_player.y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_fireball = fireball

            if closest_fireball and min_distance < 150:
                # Уклоняемся перпендикулярно траектории снаряда
                if abs(closest_fireball.vel_x) > abs(closest_fireball.vel_y):
                    # Снаряд летит горизонтально - уклоняемся вертикально
                    if self.ai_player.y > SCREEN_HEIGHT // 2:
                        self.ai_player.y = max(0, self.ai_player.y - speed * 2)
                    else:
                        self.ai_player.y = min(SCREEN_HEIGHT - PLAYER_HEIGHT, self.ai_player.y + speed * 2)
                else:
                    # Снаряд летит вертикально - уклоняемся горизонтально
                    if self.ai_player.x > SCREEN_WIDTH // 2:
                        self.ai_player.x = max(0, self.ai_player.x - speed * 2)
                    else:
                        self.ai_player.x = min(SCREEN_WIDTH - PLAYER_WIDTH, self.ai_player.x + speed * 2)
        elif action == 7:  # Ничего не делать
            pass
        elif action == 8:  # ЗАЩИТНАЯ СТРЕЛЬБА - сбивать снаряды игрока!
            # Находим ближайший снаряд игрока и стреляем по нему
            closest_enemy_fireball = None
            min_distance = float('inf')

            for fireball in self.fireballs:
                if fireball.owner == 'player':
                    distance = math.sqrt((fireball.x - (self.ai_player.x + PLAYER_WIDTH // 2)) ** 2 +
                                         (fireball.y - (self.ai_player.y + PLAYER_HEIGHT // 2)) ** 2)
                    if distance < min_distance and distance < 300:  # В пределах досягаемости
                        min_distance = distance
                        closest_enemy_fireball = fireball

            if closest_enemy_fireball and self.ai_player.can_shoot():
                # Предсказываем куда будет снаряд через несколько кадров
                prediction_time = 15 + int(self.ai.defensive_bonus * 10)
                intercept_x = closest_enemy_fireball.x + closest_enemy_fireball.vel_x * prediction_time
                intercept_y = closest_enemy_fireball.y + closest_enemy_fireball.vel_y * prediction_time

                # Стреляем на перехват
                defensive_fireball = self.ai_player.shoot(intercept_x, intercept_y)
                if defensive_fireball:
                    self.fireballs.append(defensive_fireball)
                    self.ai.shots_fired += 1

    def calculate_ai_reward(self):
        reward = 0

        # Базовая награда за выживание
        reward += 1

        # Награда за поддержание дистанции (не слишком близко, не слишком далеко)
        distance_to_player = math.sqrt(
            (self.ai_player.x - self.player.x) ** 2 + (self.ai_player.y - self.player.y) ** 2)
        if 200 <= distance_to_player <= 400:
            reward += 5
        elif distance_to_player < 100:
            reward -= 10  # Штраф за слишком близкое расстояние

        # Награда за уклонение от снарядов
        for fireball in self.fireballs:
            if fireball.owner != 'ai':
                distance_to_fireball = math.sqrt(
                    (fireball.x - self.ai_player.x) ** 2 + (fireball.y - self.ai_player.y) ** 2)
                if distance_to_fireball > 100:
                    reward += 2  # Награда за безопасную дистанцию
                elif distance_to_fireball < 50:
                    reward -= 15  # Большой штраф за опасную близость

        # Награда за эффективное использование маны
        if self.ai_player.mana > 80:
            reward += 3
        elif self.ai_player.mana < 20:
            reward -= 5

        return reward

    def update_fireballs(self):
        active_fireballs = []

        for fireball in self.fireballs:
            if fireball.update():
                active_fireballs.append(fireball)
            else:
                # Создаем взрыв на месте исчезновения снаряда
                self.explosions.append(Explosion(fireball.x, fireball.y))

        self.fireballs = active_fireballs

    def check_fireball_collisions(self):
        """Проверяет столкновения между снарядами - ИИ может сбивать ваши снаряды!"""
        fireballs_to_remove = []

        for i, fireball1 in enumerate(self.fireballs):
            for j, fireball2 in enumerate(self.fireballs):
                if i != j and fireball1.owner != fireball2.owner:  # Разные владельцы
                    distance = math.sqrt((fireball1.x - fireball2.x) ** 2 + (fireball1.y - fireball2.y) ** 2)

                    if distance < FIREBALL_SIZE * 2:  # Столкновение!
                        # Создаем взрывы на месте столкновения
                        self.explosions.append(Explosion((fireball1.x + fireball2.x) / 2,
                                                         (fireball1.y + fireball2.y) / 2))

                        # Награждаем ИИ за успешную защиту
                        if fireball1.owner == 'ai' and fireball2.owner == 'player':
                            self.ai.last_reward += 50  # Бонус за защиту
                            if self.ai.last_state is not None:
                                self.ai.update_q_table(self.ai.last_state, self.ai.last_action, 50, None)

                        # Помечаем снаряды для удаления
                        if fireball1 not in fireballs_to_remove:
                            fireballs_to_remove.append(fireball1)
                        if fireball2 not in fireballs_to_remove:
                            fireballs_to_remove.append(fireball2)

    def check_collisions(self):
        for fireball in self.fireballs[:]:
            # Попадание в игрока (снаряд ИИ попадает в игрока)
            if (fireball.owner == 'ai' and
                    fireball.x >= self.player.x - FIREBALL_SIZE and
                    fireball.x <= self.player.x + PLAYER_WIDTH + FIREBALL_SIZE and
                    fireball.y >= self.player.y - FIREBALL_SIZE and
                    fireball.y <= self.player.y + PLAYER_HEIGHT + FIREBALL_SIZE):

                if self.player.take_damage():
                    self.explosions.append(Explosion(fireball.x, fireball.y))
                    self.fireballs.remove(fireball)
                    self.ai.apply_kill_bonus()  # ИИ получает бонус за убийство игрока!

            # Попадание в ИИ (снаряд игрока попадает в ИИ)
            elif (fireball.owner == 'player' and
                  fireball.x >= self.ai_player.x - FIREBALL_SIZE and
                  fireball.x <= self.ai_player.x + PLAYER_WIDTH + FIREBALL_SIZE and
                  fireball.y >= self.ai_player.y - FIREBALL_SIZE and
                  fireball.y <= self.ai_player.y + PLAYER_HEIGHT + FIREBALL_SIZE):

                if self.ai_player.take_damage():
                    self.explosions.append(Explosion(fireball.x, fireball.y))
                    self.fireballs.remove(fireball)
                    self.ai.deaths += 1

                    # Большой штраф за смерть
                    if self.ai.last_state is not None:
                        self.ai.update_q_table(self.ai.last_state, self.ai.last_action, -100, None)

    def update_explosions(self):
        self.explosions = [explosion for explosion in self.explosions if explosion.update()]

    def draw(self):
        self.screen.fill(BLACK)

        # Рисуем игроков
        self.player.draw(self.screen)
        self.ai_player.draw(self.screen)

        # Рисуем снаряды
        for fireball in self.fireballs:
            fireball.draw(self.screen)

        # Рисуем взрывы
        for explosion in self.explosions:
            explosion.draw(self.screen)

        # Рисуем UI
        player_deaths_text = self.font.render(f"Ваши смерти: {self.player.deaths}", True, WHITE)
        ai_deaths_text = self.font.render(f"Смерти ИИ: {self.ai_player.deaths}", True, WHITE)

        self.screen.blit(player_deaths_text, (50, 30))
        self.screen.blit(ai_deaths_text, (SCREEN_WIDTH - 250, 30))

        # Статистика ИИ
        epsilon_text = self.small_font.render(f"Исследование ИИ: {self.ai.epsilon:.3f}", True, WHITE)
        accuracy_text = self.small_font.render(
            f"Точность ИИ: {(self.ai.hits_scored / (max(1, self.ai.shots_fired))) * 100:.1f}%", True, WHITE)
        states_text = self.small_font.render(f"Изучено состояний: {len(self.ai.q_table)}", True, WHITE)
        shots_text = self.small_font.render(f"Выстрелов ИИ: {self.ai.shots_fired}", True, WHITE)

        # Информация о бонусах ИИ (теперь БЕСКОНЕЧНЫХ!)
        bonuses_text = self.small_font.render(f"🏆 УБИЙСТВ: {self.ai.kill_bonuses}", True, YELLOW)
        speed_text = self.small_font.render(f"💨 Скорость: {self.ai.speed_multiplier:.2f}x", True,
                                            GREEN if self.ai.speed_multiplier > 1 else WHITE)
        accuracy_bonus_text = self.small_font.render(f"🎯 Точность: +{self.ai.accuracy_bonus:.2f}", True,
                                                     GREEN if self.ai.accuracy_bonus > 0 else WHITE)
        reaction_text = self.small_font.render(f"⚡ Реакция: +{self.ai.reaction_bonus:.2f}", True,
                                               GREEN if self.ai.reaction_bonus > 0 else WHITE)
        defensive_text = self.small_font.render(f"🛡️ Защита: +{self.ai.defensive_bonus:.2f}", True,
                                                GREEN if self.ai.defensive_bonus > 0 else WHITE)

        # Имитационное обучение
        patterns_learned = len(self.ai.player_behavior_data.get('movement_patterns', []))
        reactions_learned = len(self.ai.player_behavior_data.get('reaction_times', []))
        imitation_text = self.small_font.render(f"🧠 Изучено движений: {patterns_learned}", True, PURPLE)
        reactions_text = self.small_font.render(f"⚡ Изучено реакций: {reactions_learned}", True, PURPLE)
        learning_rate_text = self.small_font.render(f"📈 Скорость обучения: {self.ai.imitation_learning_rate:.2f}", True,
                                                    PURPLE)

        self.screen.blit(epsilon_text, (50, SCREEN_HEIGHT - 200))
        self.screen.blit(accuracy_text, (50, SCREEN_HEIGHT - 180))
        self.screen.blit(states_text, (50, SCREEN_HEIGHT - 160))
        self.screen.blit(shots_text, (50, SCREEN_HEIGHT - 140))
        self.screen.blit(bonuses_text, (50, SCREEN_HEIGHT - 120))
        self.screen.blit(speed_text, (50, SCREEN_HEIGHT - 100))
        self.screen.blit(accuracy_bonus_text, (50, SCREEN_HEIGHT - 80))
        self.screen.blit(reaction_text, (50, SCREEN_HEIGHT - 60))
        self.screen.blit(defensive_text, (50, SCREEN_HEIGHT - 40))

        self.screen.blit(imitation_text, (400, SCREEN_HEIGHT - 60))
        self.screen.blit(reactions_text, (400, SCREEN_HEIGHT - 40))
        self.screen.blit(learning_rate_text, (400, SCREEN_HEIGHT - 20))
        



        # Инструкции
        controls_text = self.small_font.render("WASD - движение, ЛКМ - стрельба, R - сброс бонусов ИИ", True, WHITE)
        save_text = self.small_font.render("SPACE - сохранить модель ИИ", True, WHITE)

        self.screen.blit(controls_text, (SCREEN_WIDTH - 420, SCREEN_HEIGHT - 60))
        self.screen.blit(save_text, (SCREEN_WIDTH - 420, SCREEN_HEIGHT - 40))

        pygame.display.flip()

    def run(self):
        running = True

        while running:
            self.frame_counter += 1  # Увеличиваем счетчик кадров

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.ai.save_model()
                        print("Модель ИИ сохранена!")
                    elif event.key == pygame.K_r:
                        # Сброс бонусов ИИ (для тестирования)
                        self.ai.kill_bonuses = 0
                        self.ai.speed_multiplier = 1.0
                        self.ai.accuracy_bonus = 0.0
                        self.ai.reaction_bonus = 0.0
                        print("Бонусы ИИ сброшены!")

            self.handle_input()
            self.update_ai()

            self.player.update()
            self.ai_player.update()

            self.update_fireballs()
            self.update_explosions()
            self.check_fireball_collisions()  # Сначала проверяем столкновения снарядов
            self.check_collisions()  # Потом столкновения с игроками

            self.draw()
            self.clock.tick(60)

        self.ai.save_model()
        pygame.quit()


if __name__ == "__main__":
    game = FireballDuel()
    game.run()