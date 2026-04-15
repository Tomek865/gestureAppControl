import pygame as pg

from Game import Game
from Screen_helper import Screen_helper
from UI_settings import UI_settings

pg.init()
screen = pg.display.set_mode(UI_settings.get_screen_start_size(), pg.RESIZABLE)
screen_size = pg.display.get_window_size()
Screen_helper.set_screen(screen)
Screen_helper.set_screen_size(screen_size)
clock = pg.time.Clock()
running = True
dt = 0
min_screen_size = (700, 400)

if __name__ == "__main__":
    game = Game(mode="normal")
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.VIDEORESIZE:
                screen_size = pg.display.get_window_size()
                Screen_helper.set_screen_size(((max(screen_size[0],min_screen_size[0])),(max(screen_size[1],min_screen_size[1]))))
                screen = pg.display.set_mode(Screen_helper.get_size(), pg.RESIZABLE)
                game.on_display_resize()
        screen.fill(UI_settings.get_screen_fill_color())
        game.update()

        pg.display.flip()

        dt = clock.tick(60) / 1000

    pg.quit()
