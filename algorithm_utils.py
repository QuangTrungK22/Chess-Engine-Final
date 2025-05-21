import random
import time
from chess_engine import GameState, Move

# Piece evaluation scores
pieceScore = {"K": 0, "Q": 90, "R": 50, "B": 35, "N": 30, "p": 10}

# Position score matrices for non-king pieces
knightScore = [[1, 1, 1, 1, 1, 1, 1, 1],
               [1, 2, 2, 2, 2, 2, 2, 1],
               [1, 2, 3, 3, 3, 3, 2, 1],
               [1, 2, 3, 4, 4, 3, 2, 1],
               [1, 2, 3, 4, 4, 3, 2, 1],
               [1, 2, 3, 3, 3, 3, 2, 1],
               [1, 2, 2, 2, 2, 2, 2, 1],
               [1, 1, 1, 1, 1, 1, 1, 1]]

bishopScore = [[4, 3, 2, 1, 1, 2, 3, 4],
               [3, 4, 3, 2, 2, 3, 4, 3],
               [2, 3, 4, 3, 3, 4, 3, 2],
               [1, 2, 3, 4, 4, 3, 2, 1],
               [1, 2, 3, 4, 4, 3, 2, 1],
               [2, 3, 4, 3, 3, 4, 3, 2],
               [3, 4, 3, 2, 2, 3, 4, 3],
               [4, 3, 2, 1, 1, 2, 3, 4]]

queenScore = [[1, 1, 1, 3, 1, 1, 1, 1],
              [1, 2, 3, 3, 3, 1, 1, 1],
              [1, 4, 3, 3, 3, 4, 2, 1],
              [1, 2, 3, 3, 3, 2, 2, 1],
              [1, 2, 3, 3, 3, 2, 2, 1],
              [1, 4, 3, 3, 3, 4, 2, 1],
              [1, 2, 3, 3, 3, 1, 1, 1],
              [1, 1, 1, 3, 1, 1, 1, 1]]

rookScore = [[4, 3, 4, 4, 4, 4, 3, 4],
             [4, 4, 4, 4, 4, 4, 4, 4],
             [1, 1, 2, 3, 3, 2, 1, 1],
             [1, 2, 3, 4, 4, 3, 2, 1],
             [1, 2, 3, 4, 4, 3, 2, 1],
             [1, 1, 2, 3, 3, 2, 1, 1],
             [4, 4, 4, 4, 4, 4, 4, 4],
             [4, 3, 4, 4, 4, 4, 3, 4]]

whitePawnScore = [[8, 8, 8, 8, 8, 8, 8, 8],
                  [8, 8, 8, 8, 8, 8, 8, 8],
                  [5, 6, 6, 7, 7, 6, 6, 5],
                  [2, 3, 3, 5, 5, 3, 3, 2],
                  [1, 2, 3, 4, 4, 3, 2, 1],
                  [1, 2, 3, 3, 3, 3, 2, 1],
                  [1, 1, 1, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0]]

blackPawnScore = [[0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0, 1, 1, 1],
                  [1, 2, 3, 3, 3, 3, 2, 1],
                  [1, 2, 3, 4, 4, 3, 2, 1],
                  [2, 3, 3, 5, 5, 3, 3, 2],
                  [5, 6, 6, 7, 7, 6, 6, 5],
                  [8, 8, 8, 8, 8, 8, 8, 8],
                  [8, 8, 8, 8, 8, 8, 8, 8]]

piecePosScores = {
    'N': knightScore,
    'B': bishopScore,
    'Q': queenScore,
    'R': rookScore,
    "wp": whitePawnScore,
    "bp": blackPawnScore
}

check_mate = 100000
stale_mate = 0
MAX_DEPTH = 3

next_move = None
nodes = 0

def find_random_move(valid_moves: list) -> Move:
    """Return a random move from the list of valid moves."""
    return random.choice(valid_moves) if valid_moves else None

def find_best_move_minimax(gs: GameState, valid_moves: list, depth = MAX_DEPTH) -> Move:
    """
    Use the minimax algorithm with alpha-beta pruning to find the best move.
    """
    global MAX_DEPTH
    if depth != MAX_DEPTH: 
        MAX_DEPTH = depth

    global next_move, nodes
    next_move = None
    nodes = 0
    alpha = -check_mate
    beta = check_mate
    
    start_time = time.time()
    find_move_minimax(gs, valid_moves, depth, alpha, beta, gs.white_to_move)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} sec, nodes: {nodes}")
    return next_move

def find_move_minimax(gs: GameState, valid_moves: list, depth: int, alpha: int, beta: int, white_to_move: bool) -> int:
    global next_move, nodes
    nodes += 1
    if depth == 0 or gs.check_mate or gs.stale_mate:
        return score_board(gs)
    random.shuffle(valid_moves)
    if white_to_move:
        max_score = -check_mate
        for move in valid_moves:
            gs.make_move(move)
            score = find_move_minimax(gs, gs.get_valid_moves(), depth - 1, alpha, beta, False)
            gs.undo_move()
            if score > max_score:
                max_score = score
                if depth == MAX_DEPTH:
                    next_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_score
    else:
        min_score = check_mate
        for move in valid_moves:
            gs.make_move(move)
            score = find_move_minimax(gs, gs.get_valid_moves(), depth - 1, alpha, beta, True)
            gs.undo_move()
            if score < min_score:
                min_score = score
                if depth == MAX_DEPTH:
                    next_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_score

def score_board(gs: GameState) -> int:
    """
    Evaluate the board. Positive score favors white, negative favors black.
    """
    if gs.check_mate:
        return -check_mate if gs.white_to_move else check_mate
    elif gs.stale_mate:
        return stale_mate
    score = 0
    for r in range(8):
        for c in range(8):
            piece = gs.board[r][c]
            if piece != "--":
                pos_score = 0
                if piece[1] != "K":
                    if piece[1] == "p":
                        pos_score = piecePosScores[piece][r][c]
                    else:
                        pos_score = piecePosScores[piece[1]][r][c]
                if piece[0] == 'w':
                    score += pieceScore[piece[1]] + pos_score
                elif piece[0] == 'b':
                    score -= pieceScore[piece[1]] + pos_score
    return score
