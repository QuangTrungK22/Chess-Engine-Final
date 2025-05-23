import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import numpy as np # type: ignore
import random
from collections import deque
from chess_env import ChessEnv  # Import the modified chess environment with dense rewards
from chess_engine import Move, GameState
import wandb
import algorithm_utils as algo
from tqdm import tqdm
from dotenv import load_dotenv
import os 

load_dotenv()
# ----------------------------
# Neural Network for DQN
# ----------------------------
class ChessDQN(nn.Module):
    def __init__(self, input_dim=833, output_dim=4096, hidden_dims=[512, 256]):
        """
        A simple fully-connected network.
        Input:
          - input_dim: size of the state vector (64 squares * 13 features + 1 turn indicator = 833)
          - output_dim: number of possible actions (64 * 64 = 4096)
          - hidden_dims: list of hidden layer sizes
        """
        super(ChessDQN, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ----------------------------
# Replay Buffer for Experience Replay
# ----------------------------
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# ----------------------------
# DQN Agent
# ----------------------------
class DQNAgent:
    def __init__(self, input_dim=833, output_dim=4096, hidden_dims=[1024,1024],
                 lr=1e-5, gamma=0.99, device=torch.device("cpu"), loading="chess_dqn_model_400.pth"):
        self.device = device
        self.policy_net = ChessDQN(input_dim, output_dim, hidden_dims).to(device)
        self.target_net = ChessDQN(input_dim, output_dim, hidden_dims).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.steps_done = 0
        if loading:
            self.policy_net.load_state_dict(torch.load(loading, map_location=torch.device('cpu')))
        self.update_target()  # Initialize target network
        self.loss_fn = nn.SmoothL1Loss()
    def update_target(self, tau=0.005):
        """Copy the policy network weights into the target network."""
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def get_action(self, state, valid_actions, epsilon, current_valid_moves=None):
        """
        Choose an action using an epsilon-greedy policy over valid actions.
        - state: current state as a numpy array.
        - valid_actions: list of action indices corresponding to legal moves.
        - epsilon: exploration rate.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if random.random() < epsilon:
            if current_valid_moves is not None:
                return random.choice(current_valid_moves)
            else:
                return random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().data.numpy().flatten()
                wandb.log({"q_value/max": np.max(q_values), "q_value/min": np.min(q_values)})
            # Only consider valid actions
            valid_q = [(action, q_values[action]) for action in valid_actions]
            best_action = max(valid_q, key=lambda x: x[1])[0]
            #get index of best action in valid action
            
            if current_valid_moves is not None:
                # Ensure the action is valid
                best_action = valid_actions.index(best_action)
                best_action = current_valid_moves[best_action]
            return best_action
    
    def encode_action(self, move: Move) -> int:
        start_index = move.start_row * 8 + move.start_col
        end_index = move.end_row * 8 + move.end_col
        return start_index * 64 + end_index
    
    def inference(self, gs, current_valid_moves, epsilon =0.1):
        """
        Cần xây dựng 1 hàm inference riêng biệt cho việc dự đoán nước đi
        - State được truyền vào ở đây sẽ là gs -> phải chuyển đổi thành state vector
        - valid_actions là danh sách các nước đi hợp lệ -> cần chuyển đổi thành action index
        - epsilon là tỉ lệ khám phá

        Nhưng sẽ có sự xung đột vì gs ở đây là game state 
        còn môi trường mà ta dùng để huấn luyện là ChessEnv
        """
        piece_to_idx = {
            "--": 0,
            "wp": 1, "wN": 2, "wB": 3, "wR": 4, "wQ": 5, "wK": 6,
            "bp": 7, "bN": 8, "bB": 9, "bR": 10, "bQ": 11, "bK": 12
        }
        state = []
        for row in gs.board:
            for square in row:
                one_hot = [0] * 13
                one_hot[piece_to_idx[square]] = 1
                state.extend(one_hot)
        # Append turn indicator: 1 for white's turn, 0 for black's turn
        state.append(1 if gs.white_to_move else 0)
        state = np.array(state, dtype=np.float32)
        
        valid_actions = [self.encode_action(m) for m in current_valid_moves]
        action = self.get_action(state, valid_actions, epsilon, current_valid_moves=current_valid_moves)
        print(action)
        return action
        



    def optimize_model(self, replay_buffer: ReplayBuffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Compute Q(s,a) for the current state
        current_q = self.policy_net(states).gather(1, actions)
        # Compute max Q-value for next state from target network
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q_target * (1 - dones)
        
        loss = self.loss_fn(current_q, target_q)
        loss = torch.nan_to_num(loss)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

# ----------------------------
# Training Loop
# ----------------------------
def train_dqn(num_episodes=1000, batch_size=64, target_update=5, target_save = 400,
              epsilon_start=1.0, epsilon_end=0.2, epsilon_decay=0.999, lr = 1e-5):
    env = ChessEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(device=device, loading=None, hidden_dims=[512, 1024])
    replay_buffer = ReplayBuffer(capacity=100000)
    epsilon = epsilon_start
    episode_rewards = []
    MAX_MOVES = 1e5
    
    wandb.login(key=os.getenv('WANDB_KEY'))
    wandb.init(
    project="chess-dqn",  # change to your project name
    name="old_sc_512_1024",
    config={
        "lr": 1e-5,
        "gamma": 0.99,
        "batch_size": batch_size,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        "target_update": 5
    }
    )
    
    for i_episode in range(int(num_episodes)):
        state = env.reset()
        total_reward = 0.0
        done = False
        losses = []
        rewards = []
        move_count = 0
        # Get valid actions for the initial state from the game engine
        valid_moves = env.game.get_valid_moves()
        valid_actions = [env.move_to_action_index(m) for m in valid_moves]
        while not done and move_count < MAX_MOVES:
            action = agent.get_action(state, valid_actions, epsilon)
            move_count += 1 
            """
            Vấn đề nằm ở env.step(action) khi không kết thúc được trò chơi với biến done = True
            """
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            rewards.append(reward)
            # wandb.log({"action_reward": reward})
            wandb.log({"reward_hist": wandb.Histogram(rewards)})

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            
            # Update valid actions based on the new game state
            valid_moves = env.game.get_valid_moves()
            valid_actions = [env.move_to_action_index(m) for m in valid_moves]

            loss = agent.optimize_model(replay_buffer, batch_size)
            if loss is not None:
                losses.append(loss)

        if move_count >= MAX_MOVES and not done:
            print(f"[WARN] End the episode {i_episode} without win")
        mean_loss = np.mean(losses)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)
        if i_episode % target_update == 0:
            agent.update_target()
        if i_episode % target_update == 0:
            print(f"Episode {i_episode}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.2f}, Loss = {np.log(mean_loss)}")
            wandb.log({
            "total_reward": total_reward,
            "mean_loss": mean_loss,
            })
            results = evaluate_agent(agent, num_games=10)
            wandb.log({
            "eval/win": results["win"],
            "eval/loss": results["loss"],
            "eval/draw": results["draw"],
            "eval/illegal_move": results["illegal_move"],
            "eval/episode": i_episode
            })
        if i_episode % target_save == 0:
            torch.save(agent.policy_net.state_dict(), f"chess_dqn_model_{i_episode}.pth")
    return agent, episode_rewards

@torch.no_grad()
def evaluate_agent(agent: DQNAgent, num_games=10):
    env = ChessEnv()
    random_agent = RandomAgent()
    results = {"win": 0, "draw": 0, "loss": 0, "illegal_move": 0}

    for i in range(num_games):
        state = env.reset()
        done = False
        MAX_MOVE = 1e4
        count_move = 0

        agent_color = "white" if i % 2 == 0 else "black"
        current_player = "white"

        valid_moves = env.game.get_valid_moves()
        valid_actions = [env.move_to_action_index(m) for m in valid_moves]

        while not done and count_move < MAX_MOVE:
            if current_player == agent_color:
                action = agent.get_action(state, valid_actions, epsilon=0.0)
            else:
                action = random_agent.get_action(valid_actions)

            count_move += 1
            state, reward, done, info = env.step(action)

            valid_moves = env.game.get_valid_moves()
            valid_actions = [env.move_to_action_index(m) for m in valid_moves]

            if done:
                if info.get("win", False):
                    if agent_color == info["color_win"]:
                        results["win"] += 1
                    elif agent_color != info["win"]:
                        results["loss"] += 1
                    print(f" info {info}, agent_color: {agent_color}")
                    # env.print_board()  
                elif info.get("draw", False):
                    results["draw"] += 1
                    
                elif info.get("illegal_move", False):
                    results["illegal_move"] += 1
                else:
                    print("Unrecognized end condition:", info)
                break

            current_player = "black" if current_player == "white" else "white"

        if count_move >= MAX_MOVE:
            print(f"[WARN] Game stuck after {MAX_MOVE} moves")

    print(f"Evaluation vs RandomAgent: {results}")
    return results


class RandomAgent:
    def get_action(self, valid_actions):
        return random.choice(valid_actions)

class MinimaxAgent:
    def __init__(self, depth=2):
        self.depth = depth

    def get_action(self, valid_moves, state: ChessEnv):
        move = algo.find_best_move_minimax(gs = state.game, valid_moves=valid_moves, depth=self.depth)
        return move

# ----------------------------
# Main Entry Point
# ----------------------------
if __name__ == "__main__":
    trained_agent, rewards = train_dqn(num_episodes=1e5)
    # Optionally, save the model
