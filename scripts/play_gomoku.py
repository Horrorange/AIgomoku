import pygame
import sys
import random
from gomoku_env import GomokuEnv, PLAYER_BLACK, PLAYER_WHITE, GRID_LINES, \
    BOARD_OFFSET, CELL_SIZE, HALF_CELL, EMPTY_SPOT, \
    SCREEN_WIDTH, SCREEN_HEIGHT  # 确保这些常量可导入或在此定义

pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("人类 vs 随机AI 五子棋 (调试中)")
clock = pygame.time.Clock()
FPS = 30

env = GomokuEnv(grid_lines=GRID_LINES)
current_observation, game_info = env.reset()

human_player = PLAYER_BLACK
ai_player = PLAYER_WHITE if human_player == PLAYER_BLACK else PLAYER_BLACK

running = True
game_active = True

font_game_status = pygame.font.Font(None, 36)


def screen_pos_to_action(screen_x, screen_y, current_env):
    print(f"  DEBUG (screen_pos_to_action): 接收到屏幕坐标 screen_x={screen_x}, screen_y={screen_y}")

    # --- 修正边界检查 ---
    # 棋盘上第一个交叉点的有效点击区域下限
    lower_click_bound = BOARD_OFFSET - HALF_CELL
    # 棋盘上最后一个交叉点的坐标
    last_intersection_coord = BOARD_OFFSET + (current_env.grid_lines - 1) * CELL_SIZE
    # 棋盘上最后一个交叉点的有效点击区域上限
    upper_click_bound = last_intersection_coord + HALF_CELL

    if not (lower_click_bound <= screen_x <= upper_click_bound and
            lower_click_bound <= screen_y <= upper_click_bound):
        print(
            f"  DEBUG (screen_pos_to_action): 点击 ({screen_x},{screen_y}) 超出精确边界 [{lower_click_bound}, {upper_click_bound}].")
        return None

    grid_col = round((screen_x - BOARD_OFFSET) / CELL_SIZE)
    grid_row = round((screen_y - BOARD_OFFSET) / CELL_SIZE)
    print(f"  DEBUG (screen_pos_to_action): 计算得到逻辑行列 grid_row={grid_row}, grid_col={grid_col}")

    if 0 <= grid_row < current_env.grid_lines and 0 <= grid_col < current_env.grid_lines:
        action = grid_row * current_env.grid_lines + grid_col
        print(f"  DEBUG (screen_pos_to_action): 计算得到动作 action={action}")
        return action

    print(
        f"  DEBUG (screen_pos_to_action): 计算得到的逻辑坐标 ({grid_row},{grid_col}) 超出棋盘逻辑边界 [0-{current_env.grid_lines - 1}].")
    return None


def get_random_ai_action(current_env):
    if current_env.game_over:
        return None
    possible_actions = []
    for action_idx in range(current_env.action_space_size):
        row, col = current_env._action_to_coords(action_idx)
        if current_env._is_valid_move(row, col):
            possible_actions.append(action_idx)
    if possible_actions:
        return random.choice(possible_actions)
    return None


print(f"游戏开始！人类玩家执 {'黑棋' if human_player == PLAYER_BLACK else '白棋'}.")
if env.current_player == human_player:
    print("轮到你了。")
else:
    print("轮到AI。")

while running:
    is_human_turn_this_frame = game_active and env.current_player == human_player  # 在循环开始时确定是否是人类回合

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print("DEBUG: 检测到 QUIT 事件。")
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            print(f"DEBUG: 检测到 MOUSEBUTTONDOWN 事件，位置 {event.pos}")
            print(
                f"DEBUG: 当前状态: game_active={game_active}, env.current_player={env.current_player} (人类玩家是 {human_player})")

            if is_human_turn_this_frame:  # 使用帧开始时确定的状态
                print("DEBUG: 条件满足，进入人类玩家落子逻辑。")
                mouse_x, mouse_y = event.pos
                action = screen_pos_to_action(mouse_x, mouse_y, env)
                print(f"DEBUG: screen_pos_to_action 返回: action={action}")

                if action is not None:
                    # 可以在这里额外检查一下 move 是否有效，用于调试
                    # temp_r, temp_c = env._action_to_coords(action)
                    # is_valid_debug = env._is_valid_move(temp_r, temp_c)
                    # print(f"DEBUG: (即时检查) 动作 {action} -> ({temp_r},{temp_c}) 是否有效? {is_valid_debug}")

                    print(f"DEBUG: 人类玩家尝试动作: {action}，将调用 env.step()")
                    current_observation, reward, done, game_info = env.step(action)  # 调用step
                    print(f"DEBUG: env.step() 返回: reward={reward}, done={done}, info='{game_info.get('status', '')}'")

                    game_active = not done
                    print(f"DEBUG: game_active 更新为 {game_active}")
                else:
                    print("DEBUG: action 为 None，可能是点击位置无效或 screen_pos_to_action 问题。")
            else:
                print("DEBUG: 非人类玩家回合或游戏未激活，忽略此次点击。")

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and not game_active:
                print("DEBUG: 检测到 R 键按下，游戏非激活状态，准备重置。")
                current_observation, game_info = env.reset()
                game_active = True
                print(f"DEBUG: 游戏已重置。新游戏开始！人类玩家执 {'黑棋' if human_player == PLAYER_BLACK else '白棋'}.")
                if env.current_player == human_player:
                    print("轮到你了。")
                else:
                    print("轮到AI。")
            elif event.key == pygame.K_q:  # 按 Q 退出
                print("DEBUG: 检测到 Q 键按下，退出游戏。")
                running = False

    if game_active and env.current_player == ai_player:  # AI回合逻辑
        print("DEBUG: 进入AI回合逻辑。")
        # pygame.time.wait(500) # 模拟AI思考，可以暂时注释掉以加快调试

        ai_action = get_random_ai_action(env)
        if ai_action is not None:
            print(f"DEBUG: AI 选择动作: {ai_action}，将调用 env.step()")
            current_observation, reward, done, game_info = env.step(ai_action)
            print(f"DEBUG: AI env.step() 返回: reward={reward}, done={done}, info='{game_info.get('status', '')}'")
            game_active = not done
            if done:
                print(f"DEBUG: AI操作后游戏结束！结果: {game_info.get('status', '')}")
        else:
            print("DEBUG: AI 没有有效移动。")
            game_active = False

    env.render()

    status_bar_text = ""
    if game_active:
        status_bar_text = "你的回合" if env.current_player == human_player else "AI的回合"
    else:
        if env.winner is not None:  # 确保winner不是None
            if env.winner == 'DRAW':
                status_bar_text = "平局! 按 R 重玩"
            elif env.winner == human_player:
                status_bar_text = "你赢了! 按 R 重玩"
            elif env.winner == ai_player:
                status_bar_text = "AI赢了! 按 R 重玩"
            else:  # 例如非法操作导致另一方赢
                winner_name = "人类" if env.winner == human_player else "AI"
                status_bar_text = f"{winner_name}因对方非法操作获胜! 按 R 重玩"

        else:  # 如果 winner 是 None 但 game_active 是 False (例如AI没有棋可下)
            status_bar_text = "游戏结束! 按 R 重玩"

    turn_text_surface = font_game_status.render(status_bar_text, True, (0, 0, 200))
    screen.blit(turn_text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)

print("正在关闭游戏...")
env.close()
pygame.quit()
sys.exit()