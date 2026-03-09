import pygame as pg
import random

from Screen_helper import Screen_helper
from UI_settings import UI_settings


class Puck:
    def __init__(self):
        self.screen = Screen_helper.get_screen()
        self.screen_size = Screen_helper.get_size()
        self.puck_pos_curr = (self.screen_size[0] / 2, self.screen_size[1] / 2)
        self.puck_pos_last = (self.screen_size[0] / 2, self.screen_size[1] / 2)
        self.puck_vector_normalized = pg.Vector2(0.0, 0.0)
        self.puck_vector_len = 0.0
        self.puck_size = (
            min(self.screen_size[0], self.screen_size[1])
            * UI_settings.get_puck_size_mul()
        )
        self.max_puck_speed = 50

    def bounce_x(self):
        self.puck_vector_normalized[0] *= -1

    def bounce_y(self):
        self.puck_vector_normalized[1] *= -1

    def set_puck_vect_norm(self, new_vect):
        self.puck_vector_normalized = new_vect

    def set_puck_vect_len(self, new_vect_len):
        self.puck_vector_len = new_vect_len

    def set_puck_pos(self, new_pos):
        self.puck_pos_last = self.puck_pos_curr
        self.puck_pos_curr = new_pos

    def get_puck_size(self):
        return self.puck_size

    def get_puck_pos(self):
        return self.puck_pos_curr

    def get_puck_vect(self):
        return (self.puck_vector_normalized, self.puck_vector_len)

    def update(self):
        self.puck_pos_last = self.puck_pos_curr
        self.puck_pos_curr += self.puck_vector_normalized * self.puck_vector_len
        if self.puck_vector_len <= 0.0:
            self.puck_vector_len = 0.0
        else:
            self.puck_vector_len -= 0.5

    def reset(self, mode):
        self.puck_pos_curr = (self.screen_size[0] / 2, self.screen_size[1] / 2)
        self.puck_pos_last = (self.screen_size[0] / 2, self.screen_size[1] / 2)
        self.puck_vector_normalized = pg.Vector2(0.0, 0.0)
        self.puck_vector_len = 0.0
        if mode == "training":
            random_x = random.uniform(-1, 1)
            random_y = random.uniform(-1, 1)
            random_speed = random.uniform(0, self.max_puck_speed)
            self.puck_vector_normalized = pg.Vector2(random_x, random_y)
            self.puck_vector_len = random_speed

    def update_puck_size(self):
        self.screen_size = Screen_helper.get_size()
        self.puck_size = (
            min(self.screen_size[0], self.screen_size[1])
            * UI_settings.get_puck_size_mul()
        )

    def update_puck_pos(self):
        self.puck_pos_curr = (self.screen_size[0] / 2, self.screen_size[1] / 2)
        self.puck_pos_last = (self.screen_size[0] / 2, self.screen_size[1] / 2)

    def draw(self):
        pg.draw.circle(
            self.screen,
            UI_settings.get_puck_color(),
            self.puck_pos_curr,
            self.puck_size,
        )
