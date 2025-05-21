from agent import MinimaxAgent, RandomAgent
from chess_env import ChessEnv
from chess_engine import GameState, Move
def run_multiple_matches(agent1, agent2, num_games=10):
    env = ChessEnv()
    results = {'agent1': 0, 'agent2': 0, 'draw': 0}
    for _ in range(num_games):
        env.reset()
        done = False
        current_agent = agent1

        while not done:
            valid_moves = env.game.get_valid_moves()
            valid_actions = [env.move_to_action_index(m) for m in valid_moves]
            if not valid_actions:
                break
            action = current_agent.select_action(env)
            _, _, done, info = env.step(action)
            current_agent = agent2 if current_agent == agent1 else agent1

        winner = env.get_winner()
        if winner == 1:
            results['agent1'] += 1
        elif winner == -1:
            results['agent2'] += 1
        else:
            results['draw'] += 1

    return results
 