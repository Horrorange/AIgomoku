import torch
import numpy as np
import itertools
import random
import math  # 确保 math 已导入
import os  # VVVV 导入 os 模块用于检查文件是否存在 VVVV
from collections import deque  # 如果 ReplayBuffer 被移到这里，则需要

from gomoku_env import GomokuEnv, PLAYER_BLACK, PLAYER_WHITE
from dqn_agent import DQNAgent, REPLAY_MEMORY_CAPACITY, BATCH_SIZE, GAMMA, \
    EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE_FREQUENCY, \
    LEARNING_RATE, device, GRID_LINES, ACTION_SPACE_SIZE

# --- 训练设置 ---
NUM_EPISODES = 10000
OPTIMIZE_EVERY_N_STEPS = 100
SAVE_MODEL_EVERY_N_EPISODES = 100
RENDER_EVERY_N_EPISODES = 0  # 大规模训练时设为0


# VVVV 新增：加载模型的设置 VVVV
# 设置为你想加载的模型的路径，如果为 None 或文件不存在，则从头训练
LOAD_MODEL_PATH = "" # 示例：加载第100局保存的模型
# LOAD_MODEL_PATH = "latest_gomoku_checkpoint.pth" # 或者你可以始终保存到一个固定的文件名

# VVVV 新增：是否恢复训练统计数据和经验回放区 (通常不恢复经验回放区) VVVV
RESTORE_TRAINING_STATS = False  # 如果为True，会尝试从checkpoint加载统计列表
RESTORE_REPLAY_BUFFER = False  # 通常设为False，让经验总是新的


def train():
    print(f"开始在设备 {device} 上训练...")

    env = GomokuEnv(grid_lines=GRID_LINES)
    agent = DQNAgent()

    # 初始化或加载训练统计数据
    episode_durations = []
    losses_during_training = []
    episode_outcomes = []  # 'WIN_P1', 'WIN_P2', 'DRAW', 'ILLEGAL_WIN_P1', 'ILLEGAL_WIN_P2'

    start_episode = 0  # 默认从第0局开始

    # --- 尝试加载模型和训练状态 ---
    if LOAD_MODEL_PATH and os.path.exists(LOAD_MODEL_PATH):
        print(f"尝试从 '{LOAD_MODEL_PATH}' 加载检查点...")
        try:
            checkpoint = torch.load(LOAD_MODEL_PATH, map_location=device)

            agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            # 目标网络也应该从检查点加载，或者从策略网络同步
            if 'target_net_state_dict' in checkpoint:
                agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            else:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())  # 从policy_net同步
            agent.target_net.eval()  # 确保是评估模式

            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            start_episode = checkpoint.get('episode', 0)  # .get提供默认值以兼容旧格式
            agent.steps_done = checkpoint.get('steps_done', 0)
            agent.num_optimizations = checkpoint.get('num_optimizations', 0)

            if RESTORE_TRAINING_STATS:
                episode_durations = checkpoint.get('episode_durations', [])
                losses_during_training = checkpoint.get('losses_during_training', [])
                episode_outcomes = checkpoint.get('episode_outcomes', [])

            if RESTORE_REPLAY_BUFFER and 'replay_buffer' in checkpoint:
                # agent.memory.buffer = checkpoint['replay_buffer'] # 直接赋值 deque
                # 更安全的方式是逐条push，但这会很慢，通常不这么做
                # 或者确保ReplayBuffer类支持从保存的状态恢复
                # 为简单起见，我们这里通常不恢复buffer，或者你需要自己实现buffer的序列化和反序列化
                # 这里我们假设如果需要恢复，checkpoint['replay_buffer'] 就是一个 deque
                # 注意：如果deque的maxlen变了，直接赋值可能会有问题
                if isinstance(checkpoint['replay_buffer'], deque) and \
                        agent.memory.capacity == checkpoint['replay_buffer'].maxlen:
                    agent.memory.buffer = checkpoint['replay_buffer']
                    print(f"  经验回放区已从检查点恢复，当前大小: {len(agent.memory.buffer)}")
                else:
                    print(f"  警告: 经验回放区未从检查点恢复 (类型或容量不匹配，或未保存)。")

            print(f"模型和训练状态从 '{LOAD_MODEL_PATH}' 加载成功。")
            print(f"将从第 {start_episode + 1} 回合继续训练。")
            print(f"已加载 steps_done: {agent.steps_done}, num_optimizations: {agent.num_optimizations}")

        except Exception as e:
            print(f"加载检查点 '{LOAD_MODEL_PATH}' 失败: {e}")
            print("将从头开始训练。")
            start_episode = 0  # 确保如果加载失败，从头开始
    else:
        if LOAD_MODEL_PATH:
            print(f"未找到检查点文件 '{LOAD_MODEL_PATH}'。")
        print("从头开始训练。")

    # --- 主训练循环 ---
    # 从 start_episode 开始，到 NUM_EPISODES 结束 (总共还是训练 NUM_EPISODES 那么多局，除非你调整NUM_EPISODES的含义)
    # 如果想让NUM_EPISODES是总目标局数，循环应该是 for i_episode in range(start_episode, NUM_EPISODES):
    # 这里我们假设 NUM_EPISODES 是指“本次运行要新训练的局数”
    # 如果 start_episode > 0, 实际总局数会超过 NUM_EPISODES
    # 更常见的做法是：
    # total_target_episodes = 10000
    # for i_episode in range(start_episode, total_target_episodes):

    print(f"开始训练循环，从第 {start_episode + 1} 回合到第 {start_episode + NUM_EPISODES} 回合...")

    for i_episode_offset in range(NUM_EPISODES):
        current_episode_num = start_episode + i_episode_offset  # 当前实际的局数编号

        current_state_np, _ = env.reset()
        episode_loss_sum = 0
        num_optimizations_this_episode = 0

        for t in itertools.count():
            action_tensor = agent.select_action(current_state_np)
            action_item = action_tensor.item()
            player_making_move = env.current_player
            next_state_np, reward_float, done_bool, info_dict = env.step(action_item)
            agent.memory.push(current_state_np, action_item, reward_float, next_state_np, done_bool)
            current_state_np = next_state_np

            if len(agent.memory) > BATCH_SIZE and t % OPTIMIZE_EVERY_N_STEPS == 0:
                loss_value = agent.optimize_model()
                if loss_value is not None:
                    losses_during_training.append(loss_value)
                    episode_loss_sum += loss_value
                    num_optimizations_this_episode += 1

            if RENDER_EVERY_N_EPISODES > 0 and (current_episode_num + 1) % RENDER_EVERY_N_EPISODES == 0:
                env.render()

            if done_bool:
                episode_durations.append(t + 1)
                winner_info = info_dict.get('winner')
                status_info = info_dict.get('status', '')
                if "Illegal move" in status_info:
                    if winner_info == PLAYER_BLACK:
                        episode_outcomes.append('ILLEGAL_WIN_P1')
                    elif winner_info == PLAYER_WHITE:
                        episode_outcomes.append('ILLEGAL_WIN_P2')
                elif winner_info == PLAYER_BLACK:
                    episode_outcomes.append('WIN_P1')
                elif winner_info == PLAYER_WHITE:
                    episode_outcomes.append('WIN_P2')
                elif winner_info == 'DRAW':
                    episode_outcomes.append('DRAW')
                else:
                    episode_outcomes.append('UNKNOWN')
                break

        # 打印统计信息
        if (current_episode_num + 1) % 10 == 0:
            avg_duration = np.mean(episode_durations[-10:]) if len(episode_durations) >= 10 else np.mean(
                episode_durations) if episode_durations else 0
            recent_losses_to_average = min(len(losses_during_training), 100 * (
                BATCH_SIZE // OPTIMIZE_EVERY_N_STEPS if OPTIMIZE_EVERY_N_STEPS > 0 else BATCH_SIZE))
            avg_loss = np.mean(
                losses_during_training[-recent_losses_to_average:]) if recent_losses_to_average > 0 else float('nan')
            current_epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * agent.steps_done / EPS_DECAY)

            log_msg = (f"回合 [{current_episode_num + 1}/{start_episode + NUM_EPISODES}] - "
                       f"Epsilon: {current_epsilon:.4f} - Avg Loss: {avg_loss:.4f} - "
                       f"Avg Steps: {avg_duration:.2f} (last {min(len(episode_durations), 10)}) - "
                       f"Total Steps: {agent.steps_done} - Mem: {len(agent.memory)}")

            # 计算胜率 (例如最近100局或所有已记录局数中较小者)
            recent_outcomes_count = min(len(episode_outcomes), 100)
            if recent_outcomes_count > 0:
                recent_outcomes_for_stats = episode_outcomes[-recent_outcomes_count:]
                p1_wins = recent_outcomes_for_stats.count('WIN_P1') + recent_outcomes_for_stats.count('ILLEGAL_WIN_P1')
                p2_wins = recent_outcomes_for_stats.count('WIN_P2') + recent_outcomes_for_stats.count('ILLEGAL_WIN_P2')
                draws = recent_outcomes_for_stats.count('DRAW')
                log_msg += (f" - Rates (last {recent_outcomes_count}): "
                            f"P1Win: {p1_wins / recent_outcomes_count:.2%} | "
                            f"P2Win: {p2_wins / recent_outcomes_count:.2%} | "
                            f"Draw: {draws / recent_outcomes_count:.2%}")
            print(log_msg)

        # 保存检查点 (包含更多信息)
        if (current_episode_num + 1) % SAVE_MODEL_EVERY_N_EPISODES == 0:
            checkpoint_path = f"gomoku_dqn_checkpoint_episode_{current_episode_num + 1}.pth"
            checkpoint = {
                'episode': current_episode_num + 1,  # 保存的是下一轮开始的局数
                'steps_done': agent.steps_done,
                'num_optimizations': agent.num_optimizations,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),  # 保存目标网络以保持一致性
                'optimizer_state_dict': agent.optimizer.state_dict(),
                # 如果需要，也可以保存统计列表
                'episode_durations': episode_durations if RESTORE_TRAINING_STATS else [],
                'losses_during_training': losses_during_training if RESTORE_TRAINING_STATS else [],
                'episode_outcomes': episode_outcomes if RESTORE_TRAINING_STATS else []
            }
            # 如果你想保存经验回放区 (注意：文件会很大)
            # if RESTORE_REPLAY_BUFFER: # (或者一个专门的 SAVE_REPLAY_BUFFER 标志)
            #    checkpoint['replay_buffer'] = agent.memory.buffer

            torch.save(checkpoint, checkpoint_path)
            print(f"检查点已保存到: {checkpoint_path}")

    print(f"\n训练完成 (总共 {start_episode + NUM_EPISODES} 回合)。")
    final_model_path = f"gomoku_dqn_final_{start_episode + NUM_EPISODES}.pth"
    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    env.close()


if __name__ == '__main__':
    train()