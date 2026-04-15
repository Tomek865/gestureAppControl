import pygame as pg

from Screen_helper import Screen_helper
from UI_settings import UI_settings
from gesture_controll import HandTracker

class Player:
    def __init__(self, side="left", is_ai=False):
        self.is_ai = is_ai
        self.side = side  # "left" lub "right"
        self.screen = Screen_helper.get_screen()
        self.screen_size = Screen_helper.get_size()
        self.max_player_speed = 50

        if self.side == "left":
            start_x = 100
        else:
            start_x = 700

        self.start_pos = pg.Vector2(start_x, self.screen_size[1] // 2)

        self.player_pos_curr = self.start_pos.copy()
        self.player_pos_last = self.start_pos.copy()

        self.player_vect = pg.Vector2(0, 0)
        self.player_size = (
            min(self.screen_size[0], self.screen_size[1])
            * UI_settings.get_player_size_mul()
        )

        self.speed_limit = 15
        self.tracker = HandTracker()

    def reset(self):
        self.player_pos_curr = self.start_pos.copy()
        self.player_pos_last = self.start_pos.copy()
        self.player_vect = pg.Vector2(0, 0)

    def update_player_pos(self):
        if self.is_ai:
            return
        self.player_pos_last = self.player_pos_curr.copy()

        pos = self.tracker.get_position()

        update_vect = pg.Vector2(
            pg.Vector2(pos[0], pos[1]) - self.player_pos_last
        )
        if update_vect.length() > self.max_player_speed:
            update_vect = update_vect.normalize() * self.max_player_speed
        self.player_pos_curr = self.player_pos_last + update_vect

    def move_ai_step(self, action_x, action_y):
        if not self.is_ai:
            return
        self.player_pos_last = self.player_pos_curr.copy()
        move_vector = pg.Vector2(action_x, action_y) * self.speed_limit
        self.player_pos_curr += move_vector

    def calculate_player_vector(self):
        self.player_vect = self.player_pos_curr - self.player_pos_last

    def update_player_size(self):
        self.screen_size = Screen_helper.get_size()
        self.player_size = (
            min(self.screen_size[0], self.screen_size[1])
            * UI_settings.get_player_size_mul()
        )

    def get_player_vect(self):
        return self.player_vect

    def get_player_pos(self):
        return self.player_pos_curr

    def get_player_last_pos(self):
        return self.player_pos_last

    def set_player_pos(self, new_pos):
        self.player_pos_curr = new_pos

    def get_player_size(self):
        return self.player_size

    def draw(self):
        pg.draw.circle(
            self.screen,
            UI_settings.get_player_circle_color(),
            self.player_pos_curr,
            self.player_size,
            width=5,
        )

    def update(self):
        self.update_player_pos()
