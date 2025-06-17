import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import math

# --- 常量 ---
GRID_LINES = 15
ACTION_SPACE_SIZE = GRID_LINES * GRID_LINES
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 200000  # 对于五子棋这种复杂游戏，可能需要更大的值，例如 200000 或更高
TARGET_UPDATE_FREQUENCY = 10  # 例如每10次优化后更新一次目标网络，或者每 N 步/N 局游戏后
LEARNING_RATE = 1e-4  # 1e-4 or 5e-5 可能是个不错的起点
REPLAY_MEMORY_CAPACITY = 50000  # 对于五子棋，可能需要更大的容量，如 50000 或 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    # ... (保持不变) ...
    def __init__(self, input_channels=1, num_actions=ACTION_SPACE_SIZE):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64);
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128);
        self.conv_output_size = 128 * GRID_LINES * GRID_LINES
        self.fc1 = nn.Linear(self.conv_output_size, 512)  # 全连接层
        self.fc2 = nn.Linear(512, num_actions)  # 输出层

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)));
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)));

        x = x.view(x.size(0), -1)  # 展平卷积输出 (Batch_size, 128*15*15)
        x = F.relu(self.fc1(x))
        q_values_flat = self.fc2(x)  # 输出Q值
        return q_values_flat


class ReplayBuffer:
    # ... (保持不变) ...
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity);
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(state[None, :, :], dtype=torch.float32);
        next_state = torch.tensor(next_state[None, :, :], dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long);
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self.buffer): batch_size = len(self.buffer)
        if batch_size == 0: return None, None, None, None, None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.cat(states), torch.cat(actions), torch.cat(rewards), torch.cat(next_states), torch.cat(dones))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_channels=1, num_actions=ACTION_SPACE_SIZE):
        # ... (init 方法前半部分保持不变) ...
        print("DQNAgent 初始化开始...")
        self.input_channels = input_channels;
        self.num_actions = num_actions
        self.policy_net = DQN(input_channels, num_actions).to(device)
        self.target_net = DQN(input_channels, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        print("  策略网络和目标网络已创建并初始化。")
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        print(f"  优化器 AdamW 已创建，学习率: {LEARNING_RATE}")
        self.memory = ReplayBuffer(REPLAY_MEMORY_CAPACITY)
        print(f"  经验回放区已创建，容量: {REPLAY_MEMORY_CAPACITY}")
        self.steps_done = 0
        self.num_optimizations = 0  # VVVV 新增：记录优化次数，用于更新目标网络 VVVV
        print("DQNAgent 初始化完成。\n")

    def select_action(self, state_numpy):
        # ... (保持不变) ...
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state_numpy[None, None, :, :], dtype=torch.float32).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=device, dtype=torch.long)

        # In DQNAgent class, optimize_model method:

        # 在 DQNAgent 类的 optimize_model 方法中:
        # 在 DQNAgent 类的 optimize_model 方法中:
        # 在 DQNAgent 类的 optimize_model 方法中:
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)  # 此时 rewards 的形状是 [BATCH_SIZE]
        next_states = next_states.to(device)
        dones = dones.to(device)  # 此时 dones 的形状是 [BATCH_SIZE]

        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)

        if actions.ndim == 1:
            actions = actions.unsqueeze(1)  # actions 变为 [BATCH_SIZE, 1]

        current_q_values = self.policy_net(states).gather(1, actions)  # current_q_values 形状 [BATCH_SIZE, 1]
        # print(f"DEBUG (optimize_model): Shape of 'current_q_values' tensor: {current_q_values.shape}")
        # print(f"DEBUG (optimize_model): Shape of 'dones' tensor from buffer: {dones.shape}")  # 应为 [128]

        non_final_mask = ~dones
        non_final_next_states = next_states[non_final_mask]

        next_state_q_values = torch.zeros(BATCH_SIZE, 1, device=device)  # 形状 [BATCH_SIZE, 1]
        if non_final_next_states.size(0) > 0:
            next_state_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[
                0].detach().unsqueeze(1)

        # VVVV CRITICAL FIX: 调整 rewards 的形状 VVVV
        if rewards.ndim == 1:  # 如果 rewards 是一维的 [BATCH_SIZE]
            rewards_for_calc = rewards.unsqueeze(1)  # 将其变为 [BATCH_SIZE, 1]
        else:  # 如果 rewards 因为某种原因已经是 [BATCH_SIZE, 1]
            rewards_for_calc = rewards
        # ^^^^ CRITICAL FIX END ^^^^
        # print(f"DEBUG (optimize_model): Shape of 'rewards_for_calc': {rewards_for_calc.shape}")

        # 4. 计算期望的 Q 值 (贝尔曼目标: r + γ * max Q')
        # rewards_for_calc 形状是 [BATCH_SIZE, 1]
        # next_state_q_values 形状是 [BATCH_SIZE, 1]
        expected_q_values = rewards_for_calc + (GAMMA * next_state_q_values)
        # print(f"DEBUG (optimize_model): Shape of 'expected_q_values': {expected_q_values.shape}")

        # 5. 计算损失
        # current_q_values 和 expected_q_values 的形状现在都应该是 [BATCH_SIZE, 1]
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # 6. 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.num_optimizations += 1
        if self.num_optimizations % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_net()

    # VVVV 新增 update_target_net 方法 VVVV
    def update_target_net(self):
        """
        将策略网络 (policy_net) 的权重复制到目标网络 (target_net)。
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # print("  (Target Update) 目标网络权重已更新。") # 调试时开启
    # ^^^^ 新增方法结束 ^^^^


# ^^^^ DQNAgent 类结束 ^^^^


# --- 用于测试的代码 ---
if __name__ == '__main__':
    print("DQN 模型测试部分跳过。\n")
    print("ReplayBuffer 测试部分跳过。\n")

    print("开始测试 DQNAgent 的学习流程 (optimize_model)...")
    agent = DQNAgent()

    # 1. 填充一些经验到回放区 (至少达到 BATCH_SIZE)
    print(f"\n填充经验到回放区 (需要至少 {BATCH_SIZE} 条)...")
    for _ in range(BATCH_SIZE + 5):  # 填充比 BATCH_SIZE 多一点
        dummy_s = np.random.randint(0, 3, (GRID_LINES, GRID_LINES), dtype=np.int8)
        dummy_a = random.randrange(ACTION_SPACE_SIZE)
        dummy_r = float(random.randint(-1, 1))  # 奖励可以是 -1, 0, 1
        dummy_ns = np.random.randint(0, 3, (GRID_LINES, GRID_LINES), dtype=np.int8)
        dummy_d = random.choice([False, False, False, True])  # 让 done 有一定概率为 True
        agent.memory.push(dummy_s, dummy_a, dummy_r, dummy_ns, dummy_d)
    print(f"经验回放区当前大小: {len(agent.memory)}")

    # 2. 调用 optimize_model 几次
    print("\n调用 optimize_model() 5次:")
    if len(agent.memory) >= BATCH_SIZE:
        for i in range(5):
            print(f"  优化第 {i + 1} 次...")
            agent.optimize_model()
            # 在实际训练中，这里会打印 loss 值等信息
        print("  optimize_model 调用完成。")
    else:
        print("  经验不足，无法调用 optimize_model。")

    # 3. 测试目标网络更新 (通常在 optimize_model 内部根据频率调用)
    #    我们可以手动调用几次 optimize_model 来触发它
    print(f"\n测试目标网络更新 (TARGET_UPDATE_FREQUENCY={TARGET_UPDATE_FREQUENCY}):")
    if len(agent.memory) >= BATCH_SIZE:
        initial_target_net_param = next(agent.target_net.parameters()).clone()
        initial_policy_net_param = next(agent.policy_net.parameters()).clone()

        print(f"  调用 optimize_model {TARGET_UPDATE_FREQUENCY * 2} 次以确保目标网络更新...")
        for i in range(TARGET_UPDATE_FREQUENCY * 2):
            agent.optimize_model()  # 这会使 policy_net 的参数改变
            if (i + 1) % TARGET_UPDATE_FREQUENCY == 0:  # optimize_model 内部的计数器会触发更新
                print(f"    在第 {i + 1} 次优化后，目标网络应该已更新。")

        updated_target_net_param = next(agent.target_net.parameters()).clone()
        updated_policy_net_param = next(agent.policy_net.parameters()).clone()

        # 检查 policy_net 的参数是否已改变 (因为优化)
        if not torch.equal(initial_policy_net_param, updated_policy_net_param):
            print("  策略网络 (policy_net) 的参数已因优化而改变 (符合预期)。")
        else:
            print("  警告: 策略网络 (policy_net) 的参数在优化后未改变！")

        # 检查 target_net 的参数是否已更新为 policy_net 的新参数
        if torch.equal(updated_target_net_param, updated_policy_net_param):
            print("  目标网络 (target_net) 的参数已成功更新为策略网络的新参数 (符合预期)。")
        else:
            # 如果 TARGET_UPDATE_FREQUENCY 比较大，可能 initial_target_net 和 updated_target_net 会不一样
            if not torch.equal(initial_target_net_param, updated_target_net_param):
                print("  目标网络 (target_net) 的参数已改变 (符合预期，因为它从策略网络更新)。")
            else:
                print("  警告: _target_net 的参数未按预期更新！")
                print("  (可能是因为优化步骤太少或TARGET_UPDATE_FREQUENCY设置)")
    else:
        print("  经验不足，无法充分测试目标网络更新。")

    print("\nDQNAgent 学习流程测试结束。")