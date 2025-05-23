from dataclasses import dataclass, field
import copy
from typing import Tuple, List
from utils import diagonalDirections, kingDirections, knightDirections, straightDirections
import numpy as np # type: ignore
@dataclass(eq=False)
class Move:
    start_square: Tuple[int, int]
    end_square: Tuple[int, int]
    board: List[List[str]]
    is_enpassant_move: bool = False
    is_castle_move: bool = False
    piece_move: str = field(init=False)
    piece_captured: str = field(init=False)
    is_pawn_promotion: bool = field(init=False)
    is_capture: bool = field(init=False)
    move_id: int = field(init=False)
    
    # Static mappings for board coordinates
    rank_to_row = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
    row_to_rank = {v: k for k, v in rank_to_row.items()}
    file_to_col = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    col_to_file = {v: k for k, v in file_to_col.items()}
    
    def __post_init__(self):
        self.start_row, self.start_col = self.start_square
        self.end_row, self.end_col = self.end_square
        self.piece_move = self.board[self.start_row][self.start_col]
        self.piece_captured = self.board[self.end_row][self.end_col]
        self.is_pawn_promotion = ((self.piece_move == "wp" and self.end_row == 0) or 
                                  (self.piece_move == "bp" and self.end_row == 7))
        if self.is_enpassant_move:
            self.piece_captured = "wp" if self.piece_move == "bp" else "bp"
        self.is_capture = (self.piece_captured != "--")
        self.move_id = self.start_col * 1000 + self.start_row * 100 + self.end_col * 10 + self.end_row
    
    def __eq__(self, other):
        if isinstance(other, Move):
            return self.move_id == other.move_id
        return False
    
    def get_chess_notation(self) -> str:
        return self.get_file_rank(self.start_row, self.start_col) + self.get_file_rank(self.end_row, self.end_col)
    
    def get_file_rank(self, row: int, col: int) -> str:
        return self.col_to_file[col] + self.row_to_rank[row]
    
    def __str__(self) -> str:
        if self.is_castle_move:
            return "O-O" if self.end_col == 6 else "O-O-O"
        end_square = self.get_file_rank(self.end_row, self.end_col)
        if self.piece_move[1] == 'p':
            if self.is_capture:
                return self.col_to_file[self.start_col] + "x" + end_square
            else:
                return end_square
        move_str = self.piece_move[1]
        if self.is_capture:
            move_str += "x"
        return move_str + end_square

@dataclass
class CastleRights:
    wks: bool
    wqs: bool
    bks: bool
    bqs: bool

class GameState:
    """
    Represents the current state of the chess game.
    """
    def __init__(self):
        self.board = np.array([
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bp", "bp", "bp", "bp", "bp", "bp", "bp", "bp"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wp", "wp", "wp", "wp", "wp", "wp", "wp", "wp"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        ])
        self.move_functions = {
            'p': self._get_pawn_moves, 
            'R': self._get_rook_moves, 
            'N': self._get_knight_moves,
            'B': self._get_bishop_moves, 
            'Q': self._get_queen_moves, 
            'K': self._get_king_moves
        }
        self.white_to_move = True
        self.moves_log: List[Move] = []
        self.white_king_loc = (7, 4)
        self.black_king_loc = (0, 4)
        self.in_check = False
        self.check_mate = False
        self.stale_mate = False
        self.enpassant_possible = ()
        self.enpassant_possible_log = [self.enpassant_possible]
        self.pins = []
        self.checks = []
        self.current_castling_right = CastleRights(True, True, True, True)
        self.castle_rights_log = [CastleRights(
            self.current_castling_right.wks,
            self.current_castling_right.wqs,
            self.current_castling_right.bks,
            self.current_castling_right.bqs
        )]
    
    def make_move(self, move: Move) -> None:
        """Make the given move on the board."""
        self.board[move.start_row][move.start_col] = "--"
        self.board[move.end_row][move.end_col] = move.piece_move
        self.moves_log.append(move)
        self.white_to_move = not self.white_to_move
        if move.piece_move == "wK":
            self.white_king_loc = (move.end_row, move.end_col)
        if move.piece_move == "bK":
            self.black_king_loc = (move.end_row, move.end_col)
        # Handle pawn promotion
        if move.is_pawn_promotion:
            piece_promote = 'Q'
            self.board[move.end_row][move.end_col] = move.piece_move[0] + piece_promote
        # En passant capture
        if move.is_enpassant_move:
            self.board[move.start_row][move.end_col] = "--"
        # Update en passant possibility
        if move.piece_move[1] == 'p' and abs(move.start_row - move.end_row) == 2:
            self.enpassant_possible = ((move.start_row + move.end_row) // 2, move.end_col)
        else:
            self.enpassant_possible = ()
        self.enpassant_possible_log.append(self.enpassant_possible)
        # Update castling rights
        self.update_castle_right(move)
        self.castle_rights_log.append(CastleRights(
            self.current_castling_right.wks,
            self.current_castling_right.wqs,
            self.current_castling_right.bks,
            self.current_castling_right.bqs
        ))
        # Handle castling move
        if move.is_castle_move:
            if move.end_col - move.start_col == 2:  # kingside
                self.board[move.end_row][move.end_col - 1] = self.board[move.end_row][move.end_col + 1]
                self.board[move.end_row][move.end_col + 1] = "--"
            else:  # queenside
                self.board[move.end_row][move.end_col + 1] = self.board[move.end_row][move.end_col - 2]
                self.board[move.end_row][move.end_col - 2] = "--"
    
    def undo_move(self) -> None:
        """Undo the last move."""
        if self.moves_log:
            move = self.moves_log.pop()
            self.board[move.start_row][move.start_col] = move.piece_move
            self.board[move.end_row][move.end_col] = move.piece_captured
            self.white_to_move = not self.white_to_move
            if move.piece_move == "wK":
                self.white_king_loc = (move.start_row, move.start_col)
            if move.piece_move == "bK":
                self.black_king_loc = (move.start_row, move.start_col)
            # Undo en passant move
            if move.is_enpassant_move:
                self.board[move.end_row][move.end_col] = "--"
                self.board[move.start_row][move.end_col] = move.piece_captured
            self.enpassant_possible_log.pop()
            self.enpassant_possible = copy.deepcopy(self.enpassant_possible_log[-1])
            # Undo castling rights
            self.castle_rights_log.pop()
            self.current_castling_right = copy.deepcopy(self.castle_rights_log[-1])
            # Undo castling move
            if move.is_castle_move:
                if move.end_col - move.start_col == 2:
                    self.board[move.end_row][move.end_col + 1] = self.board[move.end_row][move.end_col - 1]
                    self.board[move.end_row][move.end_col - 1] = "--"
                else:
                    self.board[move.end_row][move.end_col - 2] = self.board[move.end_row][move.end_col + 1]
                    self.board[move.end_row][move.end_col + 1] = "--"
            self.check_mate = False
            self.stale_mate = False

    def update_castle_right(self, move: Move) -> None:
        """Update castling rights based on the move."""
        if move.piece_move == "wK":
            self.current_castling_right.wks = False
            self.current_castling_right.wqs = False
        elif move.piece_move == "bK":
            self.current_castling_right.bks = False
            self.current_castling_right.bqs = False
        elif move.piece_move == "wR":
            if move.start_row == 7:
                if move.start_col == 0:
                    self.current_castling_right.wqs = False
                elif move.start_col == 7:
                    self.current_castling_right.wks = False
        elif move.piece_move == "bR":
            if move.start_row == 0:
                if move.start_col == 0:
                    self.current_castling_right.bqs = False
                elif move.start_col == 7:
                    self.current_castling_right.bks = False
        if move.piece_captured == 'wR':
            if move.end_row == 7:
                if move.end_col == 0:
                    self.current_castling_right.wqs = False
                elif move.end_col == 7:
                    self.current_castling_right.wks = False
        elif move.piece_captured == 'bR':
            if move.end_row == 0:
                if move.end_col == 0:
                    self.current_castling_right.bqs = False
                elif move.end_col == 7:
                    self.current_castling_right.bks = False

    def get_valid_moves(self) -> List[Move]:
        """
        #Return all valid moves for the current game state.#
        moves = []
        self.in_check, self.pins, self.checks = self.check_for_pins_and_checks()
        king_row, king_col = (self.white_king_loc if self.white_to_move else self.black_king_loc)
        if self.in_check:
            if len(self.checks) == 1:
                moves = self.get_all_possible_moves()
                check = self.checks[0]
                valid_squares = []
                if self.board[check[0]][check[1]][1] == 'N':
                    valid_squares = [(check[0], check[1])]
                else:
                    for i in range(1, 8):
                        valid_sq = (king_row + check[2] * i, king_col + check[3] * i)
                        valid_squares.append(valid_sq)
                        if valid_sq == (check[0], check[1]):
                            break
                for i in range(len(moves) - 1, -1, -1):
                    if moves[i].piece_move[1] != 'K' and (moves[i].end_row, moves[i].end_col) not in valid_squares:
                        moves.pop(i)
            else:
                self._get_king_moves(king_row, king_col, moves)
        else:
            moves = self.get_all_possible_moves()
        if not moves:
            if self.in_check:
                self.check_mate = True
            else:
                self.stale_mate = True
        return moves
    """
    def get_valid_moves(self) -> List[Move]:
        """
        Trả về tất cả các nước đi hợp lệ cho trạng thái hiện tại của trò chơi.
        Thứ tự thực hiện:
        1. Xác định xem vua hiện tại có đang bị chiếu không, tìm các quân cờ đang ghim và các quân cờ đang chiếu.
        2. Tạo tất cả các nước đi có thể (pseudo-legal moves) dựa trên trạng thái chiếu.
           - Nếu đang bị chiếu: Chỉ có các nước đi giúp thoát chiếu (di chuyển vua, bắt quân chiếu, hoặc chặn đường chiếu) là hợp lệ.
           - Nếu không bị chiếu: Tạo tất cả các nước đi pseudo-legal.
        3. Nếu không có nước đi hợp lệ nào:
           - Nếu đang bị chiếu: Đó là chiếu hết (checkmate).
           - Nếu không bị chiếu: Đó là hết nước đi (stalemate - hòa cờ).
        """
        # self.in_check, self.pins, và self.checks được cập nhật bởi hàm này.
        # self.pins là danh sách các quân cờ bị ghim và hướng ghim.
        # self.checks là danh sách các quân cờ đang chiếu vua và hướng chiếu.
        self.in_check, self.pins, self.checks = self.check_for_pins_and_checks()

        king_row, king_col = self.white_king_loc if self.white_to_move else self.black_king_loc
        
        generated_moves = [] # Sẽ chứa các nước đi hợp lệ cuối cùng

        if self.in_check:
            if len(self.checks) == 1:  # Bị chiếu bởi một quân cờ duy nhất
                # Lấy tất cả các nước đi pseudo-legal (các hàm con sẽ tự xử lý pin)
                # Vua sẽ chỉ có các nước đi đến ô an toàn.
                # Các quân khác nếu bị ghim sẽ bị hạn chế di chuyển.
                possible_moves_when_checked = self.get_all_possible_moves()
                
                check = self.checks[0]  # Thông tin quân chiếu: (row, col, dir_x, dir_y từ vua đến quân chiếu)
                checking_piece_row, checking_piece_col = check[0], check[1]
                
                # Xác định các ô mà một quân cờ (không phải vua) có thể di chuyển đến để hóa giải thế chiếu
                # (bằng cách chặn đường hoặc bắt quân chiếu)
                valid_squares_for_block_or_capture = []
                if self.board[checking_piece_row][checking_piece_col][1] == 'N': # Nếu là Mã chiếu
                    valid_squares_for_block_or_capture.append((checking_piece_row, checking_piece_col)) # Chỉ có thể bắt Mã
                else: # Quân chiếu là quân trượt (Xe, Tượng, Hậu) hoặc Tốt
                    # check[2] và check[3] là hướng từ vua đến quân chiếu
                    for i in range(1, 8):
                        block_or_capture_sq = (king_row + check[2] * i, king_col + check[3] * i)
                        valid_squares_for_block_or_capture.append(block_or_capture_sq)
                        if block_or_capture_sq == (checking_piece_row, checking_piece_col): # Đã đến vị trí quân chiếu
                            break
                
                # Lọc các nước đi:
                for move in possible_moves_when_checked:
                    if move.piece_move[1] == 'K': # Nước đi của Vua (đã được _get_king_moves kiểm tra an toàn)
                        generated_moves.append(move)
                    else: # Nước đi của quân khác
                        # Quân này phải di chuyển đến một ô trong valid_squares_for_block_or_capture
                        # Hoặc nó đang bắt quân cờ đang chiếu (nếu nước đi là nước bắt quân)
                        if (move.end_row, move.end_col) in valid_squares_for_block_or_capture:
                            generated_moves.append(move)
            else:  # Bị chiếu đôi (hoặc nhiều hơn)
                # Chỉ có nước đi của Vua là hợp lệ.
                # _get_king_moves sẽ thêm các nước đi hợp lệ của vua vào generated_moves.
                self._get_king_moves(king_row, king_col, generated_moves)
        else:  # Không bị chiếu
            # Lấy tất cả các nước đi pseudo-legal.
            # Các nước nhập thành cũng được tạo trong self.get_all_possible_moves() (thông qua _get_king_moves)
            # và _get_castle_moves đã tự kiểm tra điều kiện không bị chiếu.
            generated_moves = self.get_all_possible_moves()

        # Sau khi có danh sách các nước đi hợp lệ (generated_moves),
        # xác định trạng thái chiếu hết hoặc hết nước đi (hòa cờ).
        if not generated_moves: # Nếu không có nước đi hợp lệ nào
            if self.in_check:
                self.check_mate = True
                # print("CHECKMATE determined") # Debug
            else:
                self.stale_mate = True
                # print("STALEMATE determined") # Debug
        else: # Nếu có nước đi hợp lệ
            self.check_mate = False
            self.stale_mate = False
            
        return generated_moves
    def get_all_possible_moves(self) -> List[Move]:
        """Generate all possible moves for the current player."""
        moves = []
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece[0] == ('w' if self.white_to_move else 'b'):
                    self.move_functions[piece[1]](r, c, moves)
        return moves

    def _get_pawn_moves(self, r: int, c: int, moves: List[Move]) -> None:
        """
        piece_pinned = False
        pin_direction = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                self.pins.pop(i)
                break
        king_row, king_col = (self.white_king_loc if self.white_to_move else self.black_king_loc)
        if self.white_to_move:
            if self.board[r-1][c] == "--" and (not piece_pinned or pin_direction == (-1, 0)):
                moves.append(Move((r, c), (r-1, c), self.board))
                if r == 6 and self.board[r-2][c] == "--":
                    moves.append(Move((r, c), (r-2, c), self.board))
            if c-1 >= 0:
                if self.board[r-1][c-1][0] == "b" and (not piece_pinned or pin_direction == (-1, -1)):
                    moves.append(Move((r, c), (r-1, c-1), self.board))
                elif (r-1, c-1) == self.enpassant_possible:
                    attacking_piece = blocking_piece = False
                    if king_row == r:
                        if king_col < c:
                            inside_range = range(king_col + 1, c)
                            outside_range = range(c + 1, 8)
                        else:
                            inside_range = range(king_col - 1, c, -1)
                            outside_range = range(c - 1, -1, -1)
                        for i in inside_range:
                            if self.board[r][i] != "--":
                                blocking_piece = True
                        for i in outside_range:
                            square = self.board[r][i]
                            if square[0] == 'b' and square[1] in ['R', 'Q']:
                                attacking_piece = True
                            elif square != "--":
                                blocking_piece = True
                    if not attacking_piece or blocking_piece:
                        moves.append(Move((r, c), (r-1, c-1), self.board, is_enpassant_move=True))
            if c+1 <= 7:
                if self.board[r-1][c+1][0] == "b" and (not piece_pinned or pin_direction == (-1, 1)):
                    moves.append(Move((r, c), (r-1, c+1), self.board))
                elif (r-1, c+1) == self.enpassant_possible:
                    attacking_piece = blocking_piece = False
                    if king_row == r:
                        if king_col < c:
                            inside_range = range(king_col + 1, c + 1)
                            outside_range = range(c + 2, 8)
                        else:
                            inside_range = range(king_col - 1, c + 1, -1)
                            outside_range = range(c - 1, -1, -1)
                        for i in inside_range:
                            if self.board[r][i] != "--":
                                blocking_piece = True
                        for i in outside_range:
                            square = self.board[r][i]
                            if square[0] == 'b' and square[1] in ['R', 'Q']:
                                attacking_piece = True
                            elif square != "--":
                                blocking_piece = True
                    if not attacking_piece or blocking_piece:
                        moves.append(Move((r, c), (r-1, c+1), self.board, is_enpassant_move=True))
        else:
            if self.board[r+1][c] == "--" and (not piece_pinned or pin_direction == (1, 0)):
                moves.append(Move((r, c), (r+1, c), self.board))
                if r == 1 and self.board[r+2][c] == "--":
                    moves.append(Move((r, c), (r+2, c), self.board))
            if c-1 >= 0:
                if self.board[r+1][c-1][0] == "w" and (not piece_pinned or pin_direction == (1, -1)):
                    moves.append(Move((r, c), (r+1, c-1), self.board))
                elif (r+1, c-1) == self.enpassant_possible:
                    attacking_piece = blocking_piece = False
                    if king_row == r:
                        if king_col < c:
                            inside_range = range(king_col + 1, c)
                            outside_range = range(c + 1, 8)
                        else:
                            inside_range = range(king_col - 1, c, -1)
                            outside_range = range(c - 1, -1, -1)
                        for i in inside_range:
                            if self.board[r][i] != "--":
                                blocking_piece = True
                        for i in outside_range:
                            square = self.board[r][i]
                            if square[0] == 'w' and square[1] in ['R', 'Q']:
                                attacking_piece = True
                            elif square != "--":
                                blocking_piece = True
                    if not attacking_piece or blocking_piece:
                        moves.append(Move((r, c), (r+1, c-1), self.board, is_enpassant_move=True))
            if c+1 <= 7:
                if self.board[r+1][c+1][0] == "w" and (not piece_pinned or pin_direction == (1, 1)):
                    moves.append(Move((r, c), (r+1, c+1), self.board))
                elif (r+1, c+1) == self.enpassant_possible:
                    attacking_piece = blocking_piece = False
                    if king_row == r:
                        if king_col < c:
                            inside_range = range(king_col + 1, c + 1)
                            outside_range = range(c + 2, 8)
                        else:
                            inside_range = range(king_col - 1, c + 1, -1)
                            outside_range = range(c - 1, -1, -1)
                        for i in inside_range:
                            if self.board[r][i] != "--":
                                blocking_piece = True
                        for i in outside_range:
                            square = self.board[r][i]
                            if square[0] == 'w' and square[1] in ['R', 'Q']:
                                attacking_piece = True
                            elif square != "--":
                                blocking_piece = True
                    if not attacking_piece or blocking_piece:
                        moves.append(Move((r, c), (r+1, c+1), self.board, is_enpassant_move=True))
    """
    def _get_pawn_moves(self, r: int, c: int, moves: List[Move]) -> None:
        piece_pinned = False
        pin_direction = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                # Đã xóa dòng self.pins.pop(i) từ đây
                break
        king_row, king_col = (self.white_king_loc if self.white_to_move else self.black_king_loc)
        if self.white_to_move:
            # Di chuyển một ô về phía trước
            if r-1 >= 0 and self.board[r-1][c] == "--" and (not piece_pinned or pin_direction == (-1, 0)):
                moves.append(Move((r, c), (r-1, c), self.board))
                # Di chuyển hai ô về phía trước từ vị trí ban đầu
                if r == 6 and self.board[r-2][c] == "--": # Không cần kiểm tra piece_pinned nữa vì nếu ô đầu tiên bị chặn, sẽ không đến được đây
                    moves.append(Move((r, c), (r-2, c), self.board))
            
            # Bắt quân theo đường chéo
            if c-1 >= 0: # Bắt quân ở bên trái
                if r-1 >= 0 and self.board[r-1][c-1][0] == "b" and (not piece_pinned or pin_direction == (-1, -1)):
                    moves.append(Move((r, c), (r-1, c-1), self.board))
                elif r-1 >=0 and (r-1, c-1) == self.enpassant_possible: # Bắt tốt qua đường (en passant) bên trái
                    # Kiểm tra điều kiện ghim phức tạp cho en passant:
                    # Tốt bắt và tốt bị bắt phải cùng hàng. Vua cũng phải cùng hàng đó.
                    # Giữa vua và tốt bắt không được có quân nào.
                    # Sau khi tốt bắt di chuyển, không có quân nào khác chiếu vua dọc theo hàng đó.
                    can_en_passant = False
                    if king_row == r: # Vua cùng hàng với tốt sẽ thực hiện en passant
                        # Kiểm tra xem có quân nào nằm giữa vua và tốt bắt (cột c) không
                        # và có quân nào của đối phương (Xe hoặc Hậu) có thể chiếu vua sau khi tốt di chuyển không
                        blocking_piece_between_king_and_pawn = False
                        for k_col in range(min(king_col, c) + 1, max(king_col, c)):
                            if self.board[r][k_col] != "--":
                                blocking_piece_between_king_and_pawn = True
                                break
                        
                        if not blocking_piece_between_king_and_pawn:
                            # Tạm thời thực hiện nước đi en passant để kiểm tra xem vua có bị chiếu không
                            self.board[r][c] = "--" # Tốt di chuyển
                            self.board[r][c-1] = "--" # Tốt đối phương bị bắt
                            self.board[r-1][c-1] = "wp" # Tốt đến vị trí mới
                            
                            in_check_after_enpassant, _, _ = self.check_for_pins_and_checks() # Kiểm tra vua có bị chiếu không
                            
                            # Hoàn tác nước đi tạm thời
                            self.board[r][c] = "wp"
                            self.board[r][c-1] = "bp" # Giả sử tốt đen bị bắt
                            self.board[r-1][c-1] = "--"
                            
                            if not in_check_after_enpassant:
                                can_en_passant = True
                    else: # Vua không cùng hàng, en passant không làm lộ vua trên hàng đó
                        can_en_passant = True

                    if can_en_passant and (not piece_pinned or pin_direction == (-1,-1)): # Tốt không bị ghim hoặc ghim đúng hướng
                         moves.append(Move((r,c), (r-1,c-1), self.board, is_enpassant_move=True))

            if c+1 <= 7: # Bắt quân ở bên phải
                if r-1 >= 0 and self.board[r-1][c+1][0] == "b" and (not piece_pinned or pin_direction == (-1, 1)):
                    moves.append(Move((r, c), (r-1, c+1), self.board))
                elif r-1 >=0 and (r-1, c+1) == self.enpassant_possible: # Bắt tốt qua đường (en passant) bên phải
                    can_en_passant = False
                    if king_row == r:
                        blocking_piece_between_king_and_pawn = False
                        for k_col in range(min(king_col, c) + 1, max(king_col, c)):
                            if self.board[r][k_col] != "--":
                                blocking_piece_between_king_and_pawn = True
                                break
                        if not blocking_piece_between_king_and_pawn:
                            self.board[r][c] = "--"
                            self.board[r][c+1] = "--"
                            self.board[r-1][c+1] = "wp"
                            in_check_after_enpassant, _, _ = self.check_for_pins_and_checks()
                            self.board[r][c] = "wp"
                            self.board[r][c+1] = "bp"
                            self.board[r-1][c+1] = "--"
                            if not in_check_after_enpassant:
                                can_en_passant = True
                    else:
                        can_en_passant = True
                    
                    if can_en_passant and (not piece_pinned or pin_direction == (-1,1)):
                        moves.append(Move((r,c), (r-1,c+1), self.board, is_enpassant_move=True))
        else:  # Lượt của quân đen
            # Di chuyển một ô về phía trước
            if r+1 <= 7 and self.board[r+1][c] == "--" and (not piece_pinned or pin_direction == (1, 0)):
                moves.append(Move((r, c), (r+1, c), self.board))
                # Di chuyển hai ô về phía trước từ vị trí ban đầu
                if r == 1 and self.board[r+2][c] == "--":
                    moves.append(Move((r, c), (r+2, c), self.board))

            # Bắt quân theo đường chéo
            if c-1 >= 0: # Bắt quân ở bên trái (theo hướng của quân đen)
                if r+1 <=7 and self.board[r+1][c-1][0] == "w" and (not piece_pinned or pin_direction == (1, -1)):
                    moves.append(Move((r, c), (r+1, c-1), self.board))
                elif r+1 <=7 and (r+1, c-1) == self.enpassant_possible: # Bắt tốt qua đường (en passant)
                    can_en_passant = False
                    if king_row == r:
                        blocking_piece_between_king_and_pawn = False
                        for k_col in range(min(king_col, c) + 1, max(king_col, c)):
                            if self.board[r][k_col] != "--":
                                blocking_piece_between_king_and_pawn = True
                                break
                        if not blocking_piece_between_king_and_pawn:
                            self.board[r][c] = "--"
                            self.board[r][c-1] = "--"
                            self.board[r+1][c-1] = "bp"
                            in_check_after_enpassant, _, _ = self.check_for_pins_and_checks()
                            self.board[r][c] = "bp"
                            self.board[r][c-1] = "wp" 
                            self.board[r+1][c-1] = "--"
                            if not in_check_after_enpassant:
                                can_en_passant = True
                    else:
                        can_en_passant = True

                    if can_en_passant and (not piece_pinned or pin_direction == (1,-1)):
                        moves.append(Move((r,c), (r+1, c-1), self.board, is_enpassant_move=True))

            if c+1 <= 7: # Bắt quân ở bên phải (theo hướng của quân đen)
                if r+1 <=7 and self.board[r+1][c+1][0] == "w" and (not piece_pinned or pin_direction == (1, 1)):
                    moves.append(Move((r, c), (r+1, c+1), self.board))
                elif r+1 <=7 and (r+1, c+1) == self.enpassant_possible: # Bắt tốt qua đường (en passant)
                    can_en_passant = False
                    if king_row == r:
                        blocking_piece_between_king_and_pawn = False
                        for k_col in range(min(king_col, c) + 1, max(king_col, c)):
                            if self.board[r][k_col] != "--":
                                blocking_piece_between_king_and_pawn = True
                                break
                        if not blocking_piece_between_king_and_pawn:
                            self.board[r][c] = "--"
                            self.board[r][c+1] = "--"
                            self.board[r+1][c+1] = "bp"
                            in_check_after_enpassant, _, _ = self.check_for_pins_and_checks()
                            self.board[r][c] = "bp"
                            self.board[r][c+1] = "wp"
                            self.board[r+1][c+1] = "--"
                            if not in_check_after_enpassant:
                                can_en_passant = True
                    else:
                        can_en_passant = True
                    
                    if can_en_passant and (not piece_pinned or pin_direction == (1,1)):
                        moves.append(Move((r,c), (r+1,c+1), self.board, is_enpassant_move=True))                    

    def _get_rook_moves(self, r: int, c: int, moves: List[Move]) -> None:
        piece_pinned = False
        pin_direction = ()
        # Kiểm tra xem quân Xe này có bị ghim không
        # QUAN TRỌNG: Không được xóa (pop) phần tử khỏi self.pins ở đây
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                # Đã xóa dòng: if self.board[r][c][1] != 'Q': self.pins.pop(i)
                break
        
        enemy_color = 'b' if self.white_to_move else 'w'
        
        # Các hướng di chuyển của quân Xe (thẳng)
        # straightDirections được giả định là đã định nghĩa, ví dụ: ((-1, 0), (1, 0), (0, -1), (0, 1))
        for d in straightDirections: # Duyệt qua các hướng: lên, xuống, trái, phải
            for i in range(1, 8): # Di chuyển tối đa 7 ô theo mỗi hướng
                end_row = r + d[0] * i
                end_col = c + d[1] * i
                
                if 0 <= end_row < 8 and 0 <= end_col < 8: # Kiểm tra ô đích có nằm trong bàn cờ không
                    # Nếu quân Xe không bị ghim, HOẶC nếu bị ghim nhưng di chuyển dọc theo đường ghim
                    if not piece_pinned or pin_direction == d or pin_direction == (-d[0], -d[1]):
                        end_piece = self.board[end_row][end_col] # Quân cờ ở ô đích
                        
                        if end_piece == "--":  # Ô trống
                            moves.append(Move((r, c), (end_row, end_col), self.board))
                        elif end_piece[0] == enemy_color:  # Ô có quân địch
                            moves.append(Move((r, c), (end_row, end_col), self.board))
                            break  # Không thể đi xuyên qua quân địch
                        else:  # Ô có quân mình
                            break  # Không thể đi vào ô có quân mình
                    else: # Bị ghim và cố gắng di chuyển ra khỏi đường ghim -> không được phép
                        break 
                else:  # Ra ngoài bàn cờ
                    break

    def _get_knight_moves(self, r: int, c: int, moves: List[Move]) -> None:
        piece_pinned = False
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True

                break
        ally_color = 'w' if self.white_to_move else 'b'
        for d in knightDirections:
            end_row = r + d[0]
            end_col = c + d[1]
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                if not piece_pinned:
                    end_piece = self.board[end_row][end_col]
                    if end_piece[0] != ally_color:
                        moves.append(Move((r, c), (end_row, end_col), self.board))

    def _get_bishop_moves(self, r: int, c: int, moves: List[Move]) -> None:
        piece_pinned = False
        pin_direction = ()
        # Kiểm tra xem quân Tượng này có bị ghim không
        # QUAN TRỌNG: Không được xóa (pop) phần tử khỏi self.pins ở đây
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piece_pinned = True
                pin_direction = (self.pins[i][2], self.pins[i][3])
                # Đã xóa dòng self.pins.pop(i) từ đây
                break
        
        enemy_color = 'b' if self.white_to_move else 'w'
        
        # Các hướng di chuyển của quân Tượng (chéo)
        # diagonalDirections được giả định là đã định nghĩa, ví dụ: ((-1, -1), (-1, 1), (1, -1), (1, 1))
        for d in diagonalDirections: # Duyệt qua các hướng chéo
            for i in range(1, 8): # Di chuyển tối đa 7 ô theo mỗi hướng
                end_row = r + d[0] * i
                end_col = c + d[1] * i
                
                if 0 <= end_row < 8 and 0 <= end_col < 8: # Kiểm tra ô đích có nằm trong bàn cờ không
                    # Nếu quân Tượng không bị ghim, HOẶC nếu bị ghim nhưng di chuyển dọc theo đường ghim
                    if not piece_pinned or pin_direction == d or pin_direction == (-d[0], -d[1]):
                        end_piece = self.board[end_row][end_col] # Quân cờ ở ô đích
                        
                        if end_piece == "--":  # Ô trống
                            moves.append(Move((r, c), (end_row, end_col), self.board))
                        elif end_piece[0] == enemy_color:  # Ô có quân địch
                            moves.append(Move((r, c), (end_row, end_col), self.board))
                            break  # Không thể đi xuyên qua quân địch
                        else:  # Ô có quân mình
                            break  # Không thể đi vào ô có quân mình
                    else: # Bị ghim và cố gắng di chuyển ra khỏi đường ghim -> không được phép
                        break
                else:  # Ra ngoài bàn cờ
                    break

    def _get_queen_moves(self, r: int, c: int, moves: List[Move]) -> None:
        self._get_rook_moves(r, c, moves)
        self._get_bishop_moves(r, c, moves)

    def _get_king_moves(self, r: int, c: int, moves: List[Move]) -> None:
        ally_color = 'w' if self.white_to_move else 'b'
        for i in range(8):
            end_row = r + kingDirections[i][0]
            end_col = c + kingDirections[i][1]
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                if end_piece[0] != ally_color:
                    # Temporarily move king to check for checks
                    if ally_color == 'w':
                        self.white_king_loc = (end_row, end_col)
                    else:
                        self.black_king_loc = (end_row, end_col)
                    in_check, _, _ = self.check_for_pins_and_checks()
                    if not in_check:
                        moves.append(Move((r, c), (end_row, end_col), self.board))
                    if ally_color == 'w':
                        self.white_king_loc = (r, c)
                    else:
                        self.black_king_loc = (r, c)
        self._get_castle_moves(r, c, moves, ally_color)

    def check_for_pins_and_checks(self):
        pins = []
        checks = []
        in_check = False
        if self.white_to_move:
            enemy_color = 'b'
            ally_color = 'w'
            start_row, start_col = self.white_king_loc
        else:
            enemy_color = 'w'
            ally_color = 'b'
            start_row, start_col = self.black_king_loc
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
        for j, d in enumerate(directions):
            possible_pin = ()
            for i in range(1, 8):
                end_row = start_row + d[0] * i
                end_col = start_col + d[1] * i
                if 0 <= end_row < 8 and 0 <= end_col < 8:
                    end_piece = self.board[end_row][end_col]
                    if end_piece[0] == ally_color and end_piece[1] != 'K':
                        if not possible_pin:
                            possible_pin = (end_row, end_col, d[0], d[1])
                        else:
                            break
                    elif end_piece[0] == enemy_color:
                        piece_type = end_piece[1]
                        if (0 <= j <= 3 and piece_type == 'R') or \
                           (4 <= j <= 7 and piece_type == 'B') or \
                           (i == 1 and piece_type == 'p' and ((enemy_color == 'b' and 4 <= j <= 5) or (enemy_color == 'w' and 6 <= j <= 7))) or \
                           (piece_type == 'Q') or (i == 1 and piece_type == 'K'):
                            if not possible_pin:
                                in_check = True
                                checks.append((end_row, end_col, d[0], d[1]))
                                break
                            else:
                                pins.append(possible_pin)
                                break
                        else:
                            break
                else:
                    break
        for m in knightDirections:
            end_row = start_row + m[0]
            end_col = start_col + m[1]
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                end_piece = self.board[end_row][end_col]
                if end_piece[0] == enemy_color and end_piece[1] == 'N':
                    in_check = True
                    checks.append((end_row, end_col, m[0], m[1]))
        return in_check, pins, checks

    def _get_castle_moves(self, r: int, c: int, moves: List[Move], ally_color: str) -> None:
        if self.in_check:
            return
        if (self.white_to_move and self.current_castling_right.wks) or (not self.white_to_move and self.current_castling_right.bks):
            self._get_kingside_castle_move(r, c, moves, ally_color)
        if (self.white_to_move and self.current_castling_right.wqs) or (not self.white_to_move and self.current_castling_right.bqs):
            self._get_queenside_castle_move(r, c, moves, ally_color)

    def _get_kingside_castle_move(self, r: int, c: int, moves: List[Move], ally_color: str) -> None:
        if self.board[r][c+1] == "--" and self.board[r][c+2] == "--":
            if ally_color == 'w':
                self.white_king_loc = (r, c+1)
            else:
                self.black_king_loc = (r, c+1)
            in_check1, _, _ = self.check_for_pins_and_checks()
            if ally_color == 'w':
                self.white_king_loc = (r, c+2)
            else:
                self.black_king_loc = (r, c+2)
            in_check2, _, _ = self.check_for_pins_and_checks()
            if ally_color == 'w':
                self.white_king_loc = (r, c)
            else:
                self.black_king_loc = (r, c)
            if not in_check1 and not in_check2:
                moves.append(Move((r, c), (r, c+2), self.board, is_castle_move=True))

    def _get_queenside_castle_move(self, r: int, c: int, moves: List[Move], ally_color: str) -> None:
        if self.board[r][c-1] == "--" and self.board[r][c-2] == "--" and self.board[r][c-3] == "--":
            if ally_color == 'w':
                self.white_king_loc = (r, c-1)
            else:
                self.black_king_loc = (r, c-1)
            in_check1, _, _ = self.check_for_pins_and_checks()
            if ally_color == 'w':
                self.white_king_loc = (r, c-2)
            else:
                self.black_king_loc = (r, c-2)
            in_check2, _, _ = self.check_for_pins_and_checks()
            if ally_color == 'w':
                self.white_king_loc = (r, c-3)
            else:
                self.black_king_loc = (r, c-3)
            in_check3, _, _ = self.check_for_pins_and_checks()
            if ally_color == 'w':
                self.white_king_loc = (r, c)
            else:
                self.black_king_loc = (r, c)
            if not in_check1 and not in_check2 and not in_check3:
                moves.append(Move((r, c), (r, c-2), self.board, is_castle_move=True))
