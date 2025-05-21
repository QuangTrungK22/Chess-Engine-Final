import numpy as np # type: ignore
import algorithm_utils  # for score_board
from chess_engine import GameState, Move

class ChessEnv:
    """
    A Gym-like environment wrapper for the chess engine that uses dense rewards.
    Dense reward is computed as the difference in the board evaluation score
    before and after the move.
    """
    def __init__(self):
        self.action_space = 64 * 64  # 4096 possible moves
        self.reset()
    
    def reset(self):
        self.game = GameState()
        return self._get_state_vector()
    
    def _get_state_vector(self):
        """
        Encodes the current board state into a flat vector.
        Each square is one-hot encoded with 13 features:
          0: empty, 1-6: white pieces, 7-12: black pieces.
        An extra feature indicates whose turn it is.
        """
        piece_to_idx = {
            "--": 0,
            "wp": 1, "wN": 2, "wB": 3, "wR": 4, "wQ": 5, "wK": 6,
            "bp": 7, "bN": 8, "bB": 9, "bR": 10, "bQ": 11, "bK": 12
        }
        state = []
        for row in self.game.board:
            for square in row:
                one_hot = [0] * 13
                one_hot[piece_to_idx[square]] = 1
                state.extend(one_hot)
        # Append turn indicator: 1 for white's turn, 0 for black's turn
        state.append(1 if self.game.white_to_move else 0)

        """
        In ra bàn cờ để kiểm tra xem quá trình mô phỏng có kẹt ở đâu không?\
        
        - uncomment de xem
        """
        # board = []
        # for row in self.game.board:
        #     row_squares = []
        #     for square in row:
        #         row_squares.append(square)
        #     board.append(row_squares)
        
        # # In bàn cờ dưới dạng lưới 8x8
        # print("\nChess Board:")
        # for i, row in enumerate(board):
        #     print(f"{' '.join(f'{square:>2}' for square in row)} ")
        # print(f"Turn: {'White' if self.game.white_to_move else 'Black'}\n")

        return np.array(state, dtype=np.float32)
    
    def move_to_action_index(self, move: Move) -> int:
        """
        Encodes a move as an integer in [0, 4095] based on starting and ending squares.
        """
        start_index = move.start_row * 8 + move.start_col
        end_index = move.end_row * 8 + move.end_col
        return start_index * 64 + end_index

    def decode_action(self, action_idx: int):
        """
        Decodes an integer action index into move coordinates: (start_row, start_col, end_row, end_col).
        """
        start_index = action_idx // 64
        end_index = action_idx % 64
        start_row, start_col = divmod(start_index, 8)
        end_row, end_col = divmod(end_index, 8)
        return (start_row, start_col, end_row, end_col)
    
    def step(self, action_idx: int):
        """
        Executes the action corresponding to action_idx.
        Computes the dense reward as the change in board evaluation score.
        Returns: next_state, reward, done, info
        """
        valid_moves = self.game.get_valid_moves()
        valid_moves_map = {self.move_to_action_index(m): m for m in valid_moves}
        
        if action_idx not in valid_moves_map:
            # Illegal move penalty
            reward = -0.5
            done = False
            return self._get_state_vector(), reward, done, {"illegal_move": True}
        
        # Compute the board evaluation score before the move
        old_score = algorithm_utils.score_board(self.game)
        
        move = valid_moves_map[action_idx]
        self.game.make_move(move)
        
        # Compute the board evaluation score after the move
        new_score = algorithm_utils.score_board(self.game)

        # Since the turn flips after a move, determine which side just moved:
        # If game.white_to_move is now True, then Black just moved; if False, then White just moved.
        if not self.game.white_to_move:
            # White just moved; reward is the improvement in score (new - old)
            reward = new_score - old_score
        else:
            # Black just moved; reward is the improvement for Black (old - new)
            reward = old_score - new_score
        
        """
        # khi mô phỏng lại để huấn luyện mô hình thì có 2 trường hợp xảy ra
        # 1. Trường hợp 1: Không có nước đi nào hợp lệ
        #
        # 2. Trường hợp 2: Có nước đi hợp lệ nhưng chỉ còn 2 quân cờ (Vua và Vua) 
        -> Tại sao self.game.stale_mate lại không hiệu quả?

        """
        valid_moves = self.game.get_valid_moves()
        def is_there_any_chess_piece_not_king():
            for row in self.game.board:
                for square in row:
                    if square[1] != "K" and square != "--":
                        # print(f"Found a piece: {square}")                
                        return True   
            return False
        
        """
        Có vẻ như self.game.check_mate không hoạt động như mong đợi
        tui nghĩ nên để nó tách làm 2 điều kiện riêng biệt và bổ sung điều kiện hòa cờ khi còn 2 con vua
        """
        # Check terminal conditions (if game is over, optionally add terminal bonus)
        if self.game.check_mate:
            print("Checkmate!")
            # Add a terminal bonus: +1 for win (from the perspective of the mover), -1 for loss
            reward += 50 if not self.game.white_to_move else +20
            done = True

        elif self.game.stale_mate or not is_there_any_chess_piece_not_king():
            print("Stalemate!")
            reward -= 1000
            done = True
        else:
            done = False
        
        return self._get_state_vector(), reward, done, {}

