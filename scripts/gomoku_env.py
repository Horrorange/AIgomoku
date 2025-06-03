import pygame
import sys
import numpy as np

# --- 游戏常量 ---
SCREEN_WIDTH = 670
SCREEN_HEIGHT = 670
BOARD_OFFSET = 27
CELL_SIZE = 44
GRID_LINES = 15
HALF_CELL = CELL_SIZE // 2
SCREEN_COLOR = [238, 154, 73]
LINE_COLOR = [0, 0, 0]
WHITE_COLOR = [255, 255, 255]
BLACK_COLOR = [0, 0, 0]
HIGHLIGHT_COLOR = [0, 229, 238]
EMPTY_SPOT = 0
PLAYER_BLACK = 1
PLAYER_WHITE = 2
BOARD_END_PX = BOARD_OFFSET + (GRID_LINES - 1) * CELL_SIZE

# VVVV 新增一个颜色用于高亮最后的落子 VVVV
LAST_MOVE_MARKER_COLOR = [255, 0, 0]  # 红色


class GomokuEnv:
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, grid_lines=GRID_LINES):
        print("环境初始化 (__init__) 开始...")
        self.grid_lines = grid_lines
        self.board_size = (grid_lines, grid_lines)
        self.action_space_size = self.grid_lines * self.grid_lines
        print(f"  棋盘线数: {self.grid_lines}x{self.grid_lines}")
        print(f"  动作空间大小: {self.action_space_size} (0 to {self.action_space_size - 1})")
        self.board_state = np.full(self.board_size, EMPTY_SPOT, dtype=np.int8)
        print(f"  棋盘状态数组已创建，初始值均为 {EMPTY_SPOT}")
        self.current_player = PLAYER_BLACK
        print(f"  当前玩家初始化为: PLAYER_BLACK ({PLAYER_BLACK})")
        self.game_over = False
        print(f"  游戏结束状态初始化为: {self.game_over}")
        self.winner = None

        self.screen = None
        self.clock = None
        self.font_small = None
        self.font_large = None

        # VVVV 新增属性来记录最后落子位置 VVVV
        self.last_move_coords = None  # (row, col)
        # ^^^^ 新增属性结束 ^^^^
        print("环境初始化 (__init__) 完成。\n")

    def reset(self):
        print("调用 reset() 方法...")
        self.board_state = np.full(self.board_size, EMPTY_SPOT, dtype=np.int8)
        self.current_player = PLAYER_BLACK
        self.game_over = False
        self.winner = None
        # VVVV 重置最后落子位置 VVVV
        self.last_move_coords = None
        # ^^^^ 重置结束 ^^^^
        initial_observation = self.board_state.copy()
        info = {}
        print("reset() 方法执行完毕。\n")
        return initial_observation, info

    def _action_to_coords(self, action):
        if not (0 <= action < self.action_space_size): return None, None
        row = action // self.grid_lines;  # 你原来这里有分号，我保留了
        col = action % self.grid_lines
        return row, col

    def _is_valid_move(self, row, col):
        if row is None or col is None: return False
        in_bounds = (0 <= row < self.grid_lines) and (0 <= col < self.grid_lines)
        if not in_bounds: return False
        return self.board_state[row, col] == EMPTY_SPOT

    # VVVV 修正 _check_win 方法中的循环逻辑 VVVV
    def _check_win(self, player, last_move_row, last_move_col):
        if last_move_row is None or last_move_col is None: return False
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # 检查一个方向
            for i in range(1, 5):
                r = last_move_row + dr * i
                c = last_move_col + dc * i
                if 0 <= r < self.grid_lines and 0 <= c < self.grid_lines and self.board_state[r, c] == player:
                    count += 1
                else:
                    break  # 跳出这个内层 for 循环
            # 检查相反方向
            for i in range(1, 5):
                r = last_move_row - dr * i
                c = last_move_col - dc * i
                if 0 <= r < self.grid_lines and 0 <= c < self.grid_lines and self.board_state[r, c] == player:
                    count += 1
                else:
                    break  # 跳出这个内层 for 循环
            if count >= 5:
                return True
        return False

    # ^^^^ _check_win 修正结束 ^^^^

    def _is_board_full(self):
        return not np.any(self.board_state == EMPTY_SPOT)

    def step(self, action):
        if self.game_over:
            return self.board_state.copy(), 0.0, True, {"status": "Game is already over. Call reset."}

        row, col = self._action_to_coords(action)

        if not self._is_valid_move(row, col):
            reward = -10.0;  # 分号保留
            self.game_over = True
            self.winner = PLAYER_WHITE if self.current_player == PLAYER_BLACK else PLAYER_BLACK
            info = {"status": "Illegal move", "action_coords": (row, col), "winner": self.winner}
            self.last_move_coords = None  # 非法操作，不记录最后有效落子
            return self.board_state.copy(), reward, self.game_over, info

        self.board_state[row, col] = self.current_player
        # VVVV 在成功落子后记录位置 VVVV
        self.last_move_coords = (row, col)
        # ^^^^ 记录结束 ^^^^

        reward = 0.0;  # 分号保留
        info = {}
        if self._check_win(self.current_player, row, col):
            self.game_over = True;  # 分号保留
            self.winner = self.current_player;  # 分号保留
            reward = 1.0
            info["status"] = f"Player {self.current_player} wins!";  # 分号保留
            info["winner"] = self.winner
        elif self._is_board_full():
            self.game_over = True;  # 分号保留
            self.winner = 'DRAW';  # 分号保留
            reward = 0.5
            info["status"] = "Draw!";  # 分号保留
            info["winner"] = self.winner
        else:
            self.current_player = PLAYER_WHITE if self.current_player == PLAYER_BLACK else PLAYER_BLACK
            info["status"] = f"Move successful. Next player: {self.current_player}"

        return self.board_state.copy(), reward, self.game_over, info

    def _init_pygame(self):
        if self.screen is None:
            print("  (Render) Pygame 初始化...")
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Gomoku Environment")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.font_small is None:
            self.font_small = pygame.font.Font(None, 24)
        if self.font_large is None:
            self.font_large = pygame.font.Font(None, 60)

    def _coords_to_screen(self, row, col):
        screen_x = BOARD_OFFSET + col * CELL_SIZE
        screen_y = BOARD_OFFSET + row * CELL_SIZE
        return screen_x, screen_y

    def render(self, mode='human'):
        if mode != 'human':
            return
        self._init_pygame()
        self.screen.fill(SCREEN_COLOR)

        # 2. 绘制棋盘网格和星位
        star_points_logic = [(3, 3), (11, 3), (3, 11), (11, 11), (7, 7)]
        if self.grid_lines == 19:
            star_points_logic.extend([(3, 9), (9, 3), (15, 3), (3, 15), (15, 9), (9, 15), (15, 15), (9, 9)])
        for i in range(self.grid_lines):
            start_pos_px = BOARD_OFFSET + i * CELL_SIZE
            line_width = 4 if i == 0 or i == self.grid_lines - 1 else 2
            pygame.draw.line(self.screen, LINE_COLOR, [BOARD_OFFSET, start_pos_px], [BOARD_END_PX, start_pos_px],
                             line_width)
            pygame.draw.line(self.screen, LINE_COLOR, [start_pos_px, BOARD_OFFSET], [start_pos_px, BOARD_END_PX],
                             line_width)
        for r_logic, c_logic in star_points_logic:
            if 0 <= r_logic < self.grid_lines and 0 <= c_logic < self.grid_lines:
                sp_x, sp_y = self._coords_to_screen(r_logic, c_logic)
                pygame.draw.circle(self.screen, LINE_COLOR, [sp_x, sp_y], 6, 0)

        # 3. 绘制棋子
        for r in range(self.grid_lines):
            for c in range(self.grid_lines):
                if self.board_state[r, c] != EMPTY_SPOT:
                    screen_x, screen_y = self._coords_to_screen(r, c)
                    piece_color = BLACK_COLOR if self.board_state[r, c] == PLAYER_BLACK else WHITE_COLOR
                    pygame.draw.circle(self.screen, piece_color, [screen_x, screen_y], HALF_CELL - 2, 0)

        # VVVV 新增：绘制最后落子位置的标记 VVVV
        if self.last_move_coords is not None:
            last_r, last_c = self.last_move_coords
            screen_x, screen_y = self._coords_to_screen(last_r, last_c)
            # 你可以选择标记的样式：
            # 1. 一个小的实心红点 (推荐)
            pygame.draw.circle(self.screen, LAST_MOVE_MARKER_COLOR, [screen_x, screen_y], 4, 0)  # 半径为4的红点
            # 2. 一个围绕棋子的小方框
            # marker_offset = HALF_CELL - 1
            # pygame.draw.rect(self.screen, LAST_MOVE_MARKER_COLOR,
            #                  (screen_x - marker_offset, screen_y - marker_offset,
            #                   2 * marker_offset, 2 * marker_offset), 1) # 线宽为1的方框
        # ^^^^ 绘制最后落子标记结束 ^^^^

        # 4. 显示游戏状态信息
        status_text = ""
        if self.game_over:
            if self.winner == 'DRAW':
                status_text = "DRAW!"
            elif self.winner is not None:
                winner_name = "BLACK" if self.winner == PLAYER_BLACK else "WHITE"
                status_text = f"PLAYER {winner_name} WINS!"
            img = self.font_large.render(status_text, True, (200, 0, 0))
            text_rect = img.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
            self.screen.blit(img, text_rect)
            restart_text = "Call env.reset() to play again."  # 在 play_gomoku.py 中会显示 "按 R 重玩"
            img_restart = self.font_small.render(restart_text, True, LINE_COLOR)
            restart_rect = img_restart.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            self.screen.blit(img_restart, restart_rect)
        else:
            player_name = "BLACK" if self.current_player == PLAYER_BLACK else "WHITE"
            status_text = f"Current Player: {player_name}"
            img = self.font_small.render(status_text, True, LINE_COLOR)
            self.screen.blit(img, (10, SCREEN_HEIGHT - 30))

        pygame.display.flip()
        # self.clock.tick(self.metadata['render_fps']) # 已移除

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

    # VVVV 将 close 方法正确缩进为类方法 VVVV
    def close(self):
        """关闭 Pygame 窗口并退出 Pygame。"""
        if self.screen is not None:
            print("  (Close) Shutting down Pygame...")
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font_small = None
            self.font_large = None
    # ^^^^ close 方法修正结束 ^^^^


# --- 主程序入口 (用于测试我们写的环境) ---
# (这部分保持不变，你可以用它来测试render和step)
if __name__ == '__main__':
    env = GomokuEnv()
    print("\n--- 开始测试 render() 方法 和落子标记 ---")
    obs, info = env.reset()
    env.render()
    pygame.time.wait(1000)  # 等待1秒看空棋盘

    actions_to_play = [
        7 * env.grid_lines + 7,  # 黑: (7,7) 中心
        6 * env.grid_lines + 7,  # 白: (6,7)
        7 * env.grid_lines + 8,  # 黑: (7,8)
        6 * env.grid_lines + 8,  # 白: (6,8)
    ]

    done = False
    for i, action in enumerate(actions_to_play):
        if done: break
        print(f"\nTurn {i + 1}, Player {env.current_player}, Action: {action} (-> {env._action_to_coords(action)})")
        obs, reward, done, info = env.step(action)
        env.render()
        print(f"  Reward: {reward}, Done: {done}, Info: {info}")

        running_render_pause = True;
        pause_start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - pause_start_time < 1000:  # 每步后暂停1秒
            for event_pause in pygame.event.get():
                if event_pause.type == pygame.QUIT: done = True; running_render_pause = False; break
            if not running_render_pause: break
        if done: break  # 如果在暂停期间关闭窗口

    if done and env.winner:
        print(f"\n胜利者: {env.winner}")
    elif done:
        print("\n游戏结束 (可能平局或提前退出)")

    print("\n测试渲染结束。按R键在play_gomoku.py中重置，或关闭窗口。")
    keep_window_open = True
    while keep_window_open and env.screen is not None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: keep_window_open = False
        if keep_window_open:
            env.render()
        else:
            break
    env.close()
    print("环境已关闭。")