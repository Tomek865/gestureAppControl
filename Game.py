import pygame as pg

from Board import Board
from Player import Player
from Puck import Puck
from Score import Score


class Game:
    def __init__(self, mode="normal"):
        self.board = Board()
        self.mode = mode

        if self.mode == "training":
            self.player = Player(side="right", is_ai=True)
            self.opponent = Player(side="left", is_ai=True)

        else:
            self.player = Player(side="right", is_ai=False)
            self.opponent = Player(side="left", is_ai=True)

        self.puck = Puck()
        self.score = Score()

    def run_frame_ai(self, action):
        self.player.move_ai_step(action[0], action[1])
        self._apply_boundaries(self.player, side="right")

        self._move_opponent_script()
        self._apply_boundaries(self.opponent, side="left")

        self.puck.update()
        self.puck_validation()

        if self.puck_player_collision(
            self.player.get_player_pos(), self.player.get_player_size()
        ):
            self.calculate_puck_vect_on_player_collide(self.player)

        if self.puck_player_collision(
            self.opponent.get_player_pos(), self.opponent.get_player_size()
        ):
            self.calculate_puck_vect_on_player_collide(self.opponent)

        if self.check_goal_left():
            self.score.add_point_right()
            return 1
        elif self.check_goal_right():
            self.score.add_point_left()
            return -1

        return

    def update(self):
        self.update_player()
        self._move_opponent_script()
        self._apply_boundaries(self.opponent, side="left")

        self.puck.update()

        if self.check_goal_left():
            self.score.add_point_right()
            self.reset()
            return
        elif self.check_goal_right():
            self.score.add_point_left()
            self.reset()
            return

        self.puck_validation()

        if self.puck_player_collision(
            self.player.get_player_pos(), self.player.get_player_size()
        ):
            self.calculate_puck_vect_on_player_collide(self.player)

        if self.puck_player_collision(
            self.opponent.get_player_pos(), self.opponent.get_player_size()
        ):
            self.calculate_puck_vect_on_player_collide(self.opponent)

        self.draw()

    def board_validation(self, pos, size):
        (top, bottom, left, right, _) = self.board.get_board_bounds()
        # (goal_top, goal_bottom) = self.board.get_goal_y_range()
        # goal_depth = self.board.get_goal_depth()

        p = list(pos)

        # in_goal_y = (p[1] - size > goal_top) and (p[1] + size < goal_bottom)

        min_x = left + size
        max_x = right - size

        """if in_goal_y:
            min_x = left - goal_depth + size
            max_x = right + goal_depth - size"""

        if p[0] < min_x:
            p[0] = min_x
        if p[0] > max_x:
            p[0] = max_x

        """in_left_recess = p[0] < left + size
        in_right_recess = p[0] > right - size"""

        min_y = top + size
        max_y = bottom - size

        """if in_left_recess or in_right_recess:
            min_y = goal_top + size
            max_y = goal_bottom - size"""

        if p[1] < min_y:
            p[1] = min_y
        if p[1] > max_y:
            p[1] = max_y

        return tuple(p)

    def _apply_boundaries(self, player_obj, side):
        pos = player_obj.get_player_pos()
        size = player_obj.get_player_size()

        valid_pos = self.board_validation(pos, size)

        final_pos = self.middle_line_validation(side, valid_pos)

        player_obj.set_player_pos(pg.Vector2(final_pos[0], final_pos[1]))

    def puck_validation(self):
        pos = list(self.puck.get_puck_pos())
        size = self.puck.get_puck_size()
        (top, bottom, left, right, _) = self.board.get_board_bounds()

        if pos[1] - size <= top:
            self.puck.bounce_y()
            pos[1] = top + size + 1
        elif pos[1] + size >= bottom:
            self.puck.bounce_y()
            pos[1] = bottom - size - 1

        (g_top, g_bot) = self.board.get_goal_y_range()
        is_in_goal_y = (pos[1] > g_top) and (pos[1] < g_bot)

        if not is_in_goal_y:
            if pos[0] - size <= left:
                self.puck.bounce_x()
                pos[0] = left + size + 1
            elif pos[0] + size >= right:
                self.puck.bounce_x()
                pos[0] = right - size - 1

        self.puck.set_puck_pos(tuple(pos))

    def _move_opponent_script(self):
        p_pos = self.puck.get_puck_pos()
        o_pos = self.opponent.get_player_pos()
        mid_x = self.board.middle_line_start[0] + 20
        center_y = self.board.top + (self.board.board_size[1] / 2)
        speed = 0.8
        vx, vy = 0, 0

        if p_pos[0] < mid_x:
            if p_pos[1] >= o_pos[1] + 5:
                vy = speed
            elif p_pos[1] < o_pos[1] - 5:
                vy = -speed

            if p_pos[0] >= o_pos[0] + 5:
                vx = speed
            elif p_pos[0] < o_pos[0] - 5:
                vx = -speed
        else:
            if o_pos[1] >= center_y + 5:
                vy = -speed
            elif o_pos[1] < center_y - 5:
                vy = speed

            target_x = mid_x / 4
            if o_pos[0] >= target_x + 5:
                vx = -speed
            elif o_pos[0] < target_x - 5:
                vx = speed
        self.opponent.move_ai_step(vx, vy)

    def middle_line_validation(self, side, pos):
        (_, _, _, _, middle_x) = self.board.get_board_bounds()
        p = list(pos)
        if side == "right":
            if p[0] < middle_x:
                p[0] = middle_x
        elif side == "left":
            if p[0] > middle_x:
                p[0] = middle_x
        return tuple(p)

    def update_player(self):
        self.player.update_player_pos()
        self._apply_boundaries(self.player, side="right")

    def check_goal_left(self):
        p_pos = self.puck.get_puck_pos()
        p_size = self.puck.get_puck_size()
        limit = self.board.left - self.board.get_goal_depth()
        return p_pos[0]-p_size < limit

    def check_goal_right(self):
        p_pos = self.puck.get_puck_pos()
        p_size = self.puck.get_puck_size()
        limit = self.board.right + self.board.get_goal_depth()
        return p_pos[0]+p_size > limit

    def on_display_resize(self):
        self.player.update_player_size()
        self.opponent.update_player_size()
        self.board.update_board_size()
        self.puck.update_puck_size()
        self.puck.update_puck_pos()
        self.score.update_score_size()

    def draw(self):
        self.board.draw()
        self.player.draw()
        self.opponent.draw()
        self.puck.draw()
        self.score.show_score()

    def reset(self):
        self.puck.reset(self.mode)
        self.player.reset()
        self.opponent.reset()

    def puck_player_collision(self, player_pos, player_size):
        puck_pos = self.puck.get_puck_pos()
        puck_size = self.puck.get_puck_size()
        radius_sum = player_size + puck_size
        collision_vect = pg.math.Vector2(
            puck_pos) - pg.math.Vector2(player_pos)
        dist = collision_vect.length()
        return dist < radius_sum

    def calculate_puck_vect_on_player_collide(self, player):
        puck_pos = self.puck.get_puck_pos()
        puck_vect = self.puck.get_puck_vect()
        curr_puck_vect = puck_vect[0] * puck_vect[1]
        collision_vect = pg.math.Vector2(puck_pos) - pg.math.Vector2(
            player.get_player_pos()
        )
        radius_sum = self.puck.get_puck_size() + player.get_player_size()
        dist = collision_vect.length()
        if dist <= 0:
            collision_norm = pg.math.Vector2(1, 0)
        else:
            collision_norm = collision_vect.normalize()

        penetration_depth = radius_sum - dist
        puck_pos += collision_norm * penetration_depth
        self.puck.set_puck_pos(puck_pos)
        player.calculate_player_vector()
        relative_vel = curr_puck_vect - player.get_player_vect()
        vel_along_norm = relative_vel.dot(collision_norm)
        if vel_along_norm < 0:
            restitution = 1.0
            j = -(1 + restitution) * vel_along_norm
            curr_puck_vect += j * collision_norm
        puck_speed = min(curr_puck_vect.length(), self.puck.max_puck_speed)
        if puck_speed > 0:
            puck_vect_norm = curr_puck_vect.normalize()
        else:
            puck_vect_norm = pg.math.Vector2(0, 0)
        self.puck.set_puck_vect_len(puck_speed)
        self.puck.set_puck_vect_norm(puck_vect_norm)
