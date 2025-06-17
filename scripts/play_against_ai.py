import pygame
import sys
import torch
import numpy as np
import random

# 从我们自己的文件中导入
from gomoku_env import GomokuEnv, PLAYER_BLACK, PLAYER_WHITE, GRID_LINES, \
    BOARD_OFFSET, CELL_SIZE, HALF_CELL, EMPTY_SPOT
# 确保 SCREEN_WIDTH, SCREEN_HEIGHT 在 gomoku_env.py 中是全局可访问的
# 或者在这里重新定义它们

# 从 dqn_agent.py 导入 DQN 模型类和 device 设置
from dqn_agent import DQN, ACTION_SPACE_SIZE, device  # device 会自动设为 cuda 或 cpu

# --- 模型和游戏设置 ---
MODEL_PATH = "gomoku_dqn_episode_1100.pth"  # <--- 修改这里! 指向你训练好的模型文件路径
# 例如 "gomoku_dqn_episode_1000.pth"

HUMAN_PLAYER = PLAYER_BLACK  # 人类玩家执黑，先手
# HUMAN_PLAYER = PLAYER_WHITE # 如果想让人类执白，AI先手

# --- Pygame 和环境初始化 ---
pygame.init()
pygame.font.init()

try:
    from gomoku_env import SCREEN_WIDTH, SCREEN_HEIGHT
except ImportError:
    print("警告: SCREEN_WIDTH/HEIGHT 未从 gomoku_env 导入，使用默认值 670x670")
    SCREEN_WIDTH = 670
    SCREEN_HEIGHT = 670

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(f"你 vs 训练模型 ({MODEL_PATH})")
clock = pygame.time.Clock()
FPS = 10  # 游戏帧率可以低一些，因为有人类玩家

# 创建 Gomoku 环境实例
env = GomokuEnv(grid_lines=GRID_LINES)
current_observation_np, game_info = env.reset()

# --- 加载训练好的 DQN 模型 ---
# 假设 DQN 模型的输入通道数为1 (input_channels=1)
# 这个值需要和你训练时使用的 DQN 模型定义一致
ai_model = DQN(input_channels=1, num_actions=ACTION_SPACE_SIZE).to(device)

try:
    ai_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"模型 '{MODEL_PATH}' 加载成功！")
except FileNotFoundError:
    print(f"错误: 模型文件 '{MODEL_PATH}' 未找到！请检查路径。")
    print("AI 将无法工作。")
    # 你可以选择在这里退出 sys.exit()，或者让游戏继续但AI不行动
    # 为了能继续，我们让 ai_model 保持未训练状态 (随机权重)
except Exception as e:
    print(f"加载模型时发生错误: {e}")
    print("AI 将无法工作。")

ai_model.eval()  # 设置为评估模式 (非常重要！这会关闭 dropout 和 batchnorm 的训练行为)

# --- 游戏变量 ---
ai_player = PLAYER_WHITE if HUMAN_PLAYER == PLAYER_BLACK else PLAYER_BLACK

running = True
game_active = True  # 当前这局游戏是否激活

font_game_status = pygame.font.Font(None, 30)  # 用于显示提示信息


# --- 辅助函数 ---
def screen_pos_to_action(screen_x, screen_y, current_env):
    """将屏幕像素坐标转换为棋盘格点索引 (row, col)，然后再转换为动作索引。"""
    lower_click_bound = BOARD_OFFSET - HALF_CELL
    last_intersection_coord = BOARD_OFFSET + (current_env.grid_lines - 1) * CELL_SIZE
    upper_click_bound = last_intersection_coord + HALF_CELL

    if not (lower_click_bound <= screen_x <= upper_click_bound and
            lower_click_bound <= screen_y <= upper_click_bound):
        return None

    grid_col = round((screen_x - BOARD_OFFSET) / CELL_SIZE)
    grid_row = round((screen_y - BOARD_OFFSET) / CELL_SIZE)

    if 0 <= grid_row < current_env.grid_lines and 0 <= grid_col < current_env.grid_lines:
        action = grid_row * current_env.grid_lines + grid_col
        return action
    return None


def get_trained_ai_action(board_state_np, model):
    """
    使用加载的 DQN 模型为 AI 获取动作。
    :param board_state_np: 当前棋盘状态 (NumPy 数组).
    :param model: 训练好的 PyTorch DQN 模型.
    :return: int, AI 选择的动作索引.
    """
    if env.game_over:  # 虽然主循环会检查，但这里也加一道保险
        return None

    # 1. 将 NumPy 状态转换为 PyTorch 张量，并调整形状
    #    输入状态 board_state_np: (H, W)
    #    需要转换为: (1, C, H, W) -> (1, 1, GRID_LINES, GRID_LINES)
    #    这里的 '1' 是 input_channels，需要与模型定义一致
    state_tensor = torch.tensor(board_state_np[None, None, :, :], dtype=torch.float32).to(device)

    # 2. 使用模型预测Q值 (不计算梯度)
    with torch.no_grad():
        q_values = model(state_tensor)  # 输出形状 (1, ACTION_SPACE_SIZE)

    # 3. 选择Q值最高的动作
    #    我们还需要确保这个动作是合法的。一个简单的做法是先选出Q值最高的，
    #    然后检查它是否合法。如果非法，再选择第二高的，依此类推。
    #    或者，更简单（但可能不是最优）的是直接取 argmax，让环境的 step() 去处理非法情况（返回惩罚）。
    #    对于已训练好的模型，它理论上应该学会避免非法动作。

    # 简单策略：直接选择Q值最高的动作
    # action_idx = q_values.argmax().item()

    # 稍好策略：选择Q值最高的合法动作
    # q_values[0] 的形状是 (ACTION_SPACE_SIZE)
    # argsort(-q_values[0]) 会得到按Q值降序排列的动作索引
    sorted_actions = torch.argsort(q_values[0], descending=True)

    chosen_action = None
    for action_idx_tensor in sorted_actions:
        action_idx = action_idx_tensor.item()
        r, c = env._action_to_coords(action_idx)  # 使用环境的辅助函数
        if env._is_valid_move(r, c):  # 使用环境的辅助函数
            chosen_action = action_idx
            break

    # 如果因为某种原因 (例如模型输出全是无效动作的Q值最高)，没有找到合法动作，
    # 则随机选择一个 (这不应该经常发生于训练好的模型)
    if chosen_action is None:
        print("警告: AI未能从Q值中找到有效动作，将随机选择！")
        possible_actions = []
        for i in range(ACTION_SPACE_SIZE):
            r_rand, c_rand = env._action_to_coords(i)
            if env._is_valid_move(r_rand, c_rand):
                possible_actions.append(i)
        if possible_actions:
            chosen_action = random.choice(possible_actions)
        else:  # 没有可下的地方了 (平局或已结束)
            return None

    return chosen_action


# --- 主游戏循环 ---
print(f"游戏初始化。人类玩家执 {'黑棋' if HUMAN_PLAYER == PLAYER_BLACK else '白棋'}.")
if env.current_player == HUMAN_PLAYER:
    print("轮到你了。点击棋盘落子。")
else:
    print("轮到AI。")

while running:
    # --- 事件处理 ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 鼠标左键点击
            if game_active and env.current_player == HUMAN_PLAYER:
                mouse_x, mouse_y = event.pos
                action = screen_pos_to_action(mouse_x, mouse_y, env)

                if action is not None:
                    print(f"你尝试落子在动作: {action} (坐标: {env._action_to_coords(action)})")
                    current_observation_np, reward, done, game_info = env.step(action)

                    if "Illegal move" in game_info.get("status", ""):
                        print("那是无效落子位置！AI 获胜。")
                        # env.step 内部已经处理了非法操作的后果 (game_over=True, winner=ai_player)

                    game_active = not done
                    if done:
                        print(f"游戏结束！信息: {game_info.get('status', '')}")
                else:
                    print("点击位置无效或在棋盘外。")
            elif not game_active:
                print("游戏已结束。按 'R' 重新开始。")
            elif env.current_player != HUMAN_PLAYER:
                print("还未轮到你！")

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and not game_active:  # 如果游戏结束了，按 R 键重玩
                print("按下了R键，重新开始游戏...")
                current_observation_np, game_info = env.reset()
                game_active = True
                print(f"新游戏开始！人类玩家执 {'黑棋' if HUMAN_PLAYER == PLAYER_BLACK else '白棋'}.")
                if env.current_player == HUMAN_PLAYER:
                    print("轮到你了。")
                else:
                    print("轮到AI。")
            elif event.key == pygame.K_q:  # 按 Q 退出
                running = False

    # --- AI 回合逻辑 ---
    if game_active and env.current_player == ai_player:
        print("AI 正在“思考”...")
        pygame.time.wait(100)  # 短暂延时，让玩家感知到AI在操作

        ai_action = get_trained_ai_action(current_observation_np, ai_model)

        if ai_action is not None:
            print(f"AI 选择动作: {ai_action} (坐标: {env._action_to_coords(ai_action)})")
            current_observation_np, reward, done, game_info = env.step(ai_action)
            game_active = not done
            if done:
                print(f"游戏结束！信息: {game_info.get('status', '')}")
                env.render()  # 确保渲染最后状态
                pygame.display.flip()
        else:
            # 如果AI没有有效移动 (例如棋盘满了但之前没有判断出胜负或平局)
            print("AI 没有有效移动了。游戏可能出现问题或平局。")
            game_active = False  # 结束游戏

    # --- 渲染 ---
    env.render()  # 调用环境的渲染方法来绘制棋盘、棋子和游戏结束信息

    # 在 Pygame 窗口顶部显示额外的回合/状态提示信息
    display_text = ""
    if game_active:
        if env.current_player == HUMAN_PLAYER:
            display_text = "轮到你 (点击落子)"
        else:
            display_text = "AI 正在思考..."
    else:  # 游戏结束
        winner_msg_part = ""
        if env.winner == HUMAN_PLAYER:
            winner_msg_part = "你赢了!"
        elif env.winner == ai_player:
            winner_msg_part = "AI赢了!"
        elif env.winner == 'DRAW':
            winner_msg_part = "平局!"
        else:
            winner_msg_part = f"游戏结束 ({env.winner})"  # 其他情况，如非法操作
        display_text = f"{winner_msg_part} 按 'R' 重新开始, 'Q' 退出."

    status_surface = font_game_status.render(display_text, True, (20, 20, 200))  # 深蓝色
    screen.blit(status_surface, (10, 10))  # 显示在左上角

    pygame.display.flip()  # 更新整个屏幕
    clock.tick(FPS)

# --- 清理 ---
print("正在关闭游戏...")
env.close()
pygame.quit()
sys.exit()