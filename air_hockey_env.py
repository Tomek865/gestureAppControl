import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame as pg

from Game import Game
from Screen_helper import Screen_helper
from UI_settings import UI_settings


class AirHockeyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        super(AirHockeyEnv, self).__init__()

        pg.init()
        screen_size = (800, 600)
        screen = pg.display.set_mode(screen_size)
        Screen_helper.set_screen(screen)
        Screen_helper.set_screen_size(screen_size)
        self.game = Game(mode="training")

        # AI continously controls speed < -1 ; 1 >
        # [vel_x, vel_y]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        #  [puck_x, puck_y, puck_vx, puck_vy, ai_x, ai_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

        self.clock = pg.time.Clock()
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.game.reset()
        self.current_step = 0

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # 1. ai step
        game_result = self.game.run_frame_ai(action)

        # 2. (Reward Shaping)
        reward = 0
        terminated = False
        truncated = False

        if game_result == 1:
            reward += 50.0
            terminated = True
        elif game_result == -1:
            reward -= 20.0  # later change back to -50
            terminated = True
        else:
            reward += 0.001

        puck_curr = self.game.puck.puck_pos_curr
        puck_last = self.game.puck.puck_pos_last
        player_pos_last = pg.math.Vector2(self.game.player.get_player_last_pos())
        player_pos_curr = pg.math.Vectro2(self.game.player.get_player_pos())

        if pg.math.Vector2(puck_curr).distance_to(pg.math.Vector2(puck_last)) < 0.5:
            reward -= 0.05

        if self.game.puck_player_collision(
            self.game.player.get_player_pos(), self.game.player.get_player_size()
        ):
            norm_x = self.game.puck.get_puck_vect()[0][0]
            speed = self.game.puck.get_puck_vect()[1]
            real_speed = norm_x * speed

            if real_speed < 0:
                reward += abs(real_speed) * 0.05

        w, h = Screen_helper.get_size()

        if puck_curr[0] > w / 2:
            reward -= 0.005
            old_distance = player_pos_last.distance_to(pg.math.Vector2(puck_last))
            new_distance = player_pos_curr.distance_to(pg.math.Vector2(puck_curr))
            if new_distance < old_distance:
                reward += 0.01
        elif puck_curr[0] < w / 2:
            defense_line = w * 0.75
            if self.game.player.get_player_pos()[0] < defense_line:
                reward -= 0.005

        (top, bottom, left, right, _) = self.game.board.get_board_bounds()
        size = self.game.player.get_player_size()
        max_x = right - size
        pos_x = self.game.player.get_player_pos()[0]
        pos_y = self.game.player.get_player_pos()[1]
        if pos_x >= max_x - 10:
            reward -= 0.005

        if pos_y - size <= top + 10 or pos_y + size >= bottom - 10:
            reward -= 0.005

        opponent_goal = pg.math.Vector2(0, h / 2)
        old_distance = pg.math.Vector2(puck_last).distance_to(opponent_goal)
        new_distance = pg.math.Vector2(puck_curr).distance_to(opponent_goal)

        if new_distance < old_distance:
            reward += 0.005

        if self.current_step >= self.max_steps:
            reward -= 100
            truncated = True

        observation = self._get_obs()

        return observation, reward, terminated, truncated, {}

    def render(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()

        screen = Screen_helper.get_screen()
        screen.fill(UI_settings.get_screen_fill_color())
        self.game.draw()
        pg.display.flip()

        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        pg.quit()


def _get_obs(self):
    """Pobiera i normalizuje dane dla AI (Wersja 12-parametrowa)."""
    puck = self.game.puck
    ai = self.game.player
    opp = self.game.opponent

    w, h = Screen_helper.get_size()

    # Dane krążka
    p_pos = puck.get_puck_pos()
    p_norm_vec, p_speed = puck.get_puck_vect()

    # Dane AI (Twój bot)
    ai_pos = ai.get_player_pos()
    # Obliczamy prędkość AI na podstawie różnicy pozycji (klasyczny wektor przesunięcia)
    ai_last = ai.get_player_last_pos()
    ai_vel_x = ai_pos[0] - ai_last[0]
    ai_vel_y = ai_pos[1] - ai_last[1]

    # Dane Przeciwnika (Nauczyciel)
    opp_pos = opp.get_player_pos()
    opp_last = opp.get_player_last_pos()
    opp_vel_x = opp_pos[0] - opp_last[0]
    opp_vel_y = opp_pos[1] - opp_last[1]

    obs = np.array(
        [
            float(p_pos[0]) / w,  # 1. Puck X
            float(p_pos[1]) / h,  # 2. Puck Y
            float(p_norm_vec[0] * p_speed) / 20.0,  # 3. Puck Vx
            float(p_norm_vec[1] * p_speed) / 20.0,  # 4. Puck Vy
            float(ai_pos[0]) / w,  # 5. AI X
            float(ai_pos[1]) / h,  # 6. AI Y
            float(ai_vel_x) / 15.0,  # 7. AI Vx (dzielone przez speed_limit)
            float(ai_vel_y) / 15.0,  # 8. AI Vy
            float(opp_pos[0]) / w,  # 9. Opponent X
            float(opp_pos[1]) / h,  # 10. Opponent Y
            float(opp_vel_x) / 15.0,  # 11. Opponent Vx
            float(opp_vel_y) / 15.0,  # 12. Opponent Vy
        ],
        dtype=np.float32,
    )

    return obs
