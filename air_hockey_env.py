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
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)

        #  [puck_x, puck_y, puck_vx, puck_vy, ai_x, ai_y]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
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
        # return 1 (ai win), -1 (ai lose), 0 (game continous)
        game_result = self.game.run_frame_ai(action)

        # 2. (Reward Shaping)
        reward = 0
        terminated = False
        truncated = False

        if game_result == 1:
            reward += 50.0
            terminated = True
        elif game_result == -1:
            reward -= 50.0
            terminated = True
        else:
            reward = -0.001

        # Reward for moving towards the puck using its position history
        player_pos = self.game.player.get_player_pos()
        player_pos_last = self.game.player.get_player_last_pos()
        puck_curr = self.game.puck.puck_pos_curr
        puck_last = self.game.puck.puck_pos_last

        if player_pos.distance_to(puck_curr) < player_pos.distance_to(puck_last):
            reward += 0.02

        # puck go left after collide
        puck_vect_norm_x = self.game.puck.get_puck_vect()[0][0]
        if (
            self.game.puck_player_collision(
                self.game.player.get_player_pos(), self.game.player.get_player_size()
            )
            and -1 <= puck_vect_norm_x < 0
        ):
            reward += 0.3

        if puck_curr == puck_last:
            reward -= 0.5

        if player_pos_last == player_pos:
            reward -= 1

        (top, bottom, left, right, _) = self.game.board.get_board_bounds()
        size = self.game.player.get_player_size()
        max_x = right - size
        if self.game.player.get_player_pos()[0] == max_x:
            reward -= 0.05

        # 3. stuck safety (Truncation)
        if self.current_step >= self.max_steps:
            truncated = True
            # punish for not doing anything
            # reward -= 5

        # 4. observation after move
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
        """Pobiera i normalizuje dane dla AI."""
        puck = self.game.puck
        ai = self.game.player
        opp = self.game.opponent

        w, h = Screen_helper.get_size()

        p_pos = puck.get_puck_pos()
        p_vel = puck.get_puck_vect()
        ai_pos = ai.get_player_pos()
        opp_pos = opp.get_player_pos()

        obs = np.array(
            [
                float(p_pos[0]) / w,  # Puck X
                float(p_pos[1]) / h,  # Puck Y
                float(p_vel[0][0]) / 20.0,  # Puck Vx
                float(p_vel[0][1]) / 20.0,  # Puck Vy
                float(ai_pos[0]) / w,  # AI X
                float(ai_pos[1]) / h,  # AI Y
                float(opp_pos[0]) / w,  # opponent X
                float(opp_pos[1]) / h,  # opponent Y
            ],
            dtype=np.float32,
        )

        return obs
