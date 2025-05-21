import math
import os
import pygame as p  # type: ignore
import pygame_gui
import chess_engine
from chess_engine import Move, GameState
import algorithm_utils

#========================
from agent import DQNAgent
import torch
#=======================

# from main import animateMove, highlight_move,draw_moveslog
# Constants
WIDTH = HEIGHT = 512
MOVE_LOG_PANEL_WIDTH = 290
MOVE_LOG_PANEL_HEIGHT = HEIGHT
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 60
IMAGES = {}

# UI layout
BUTTON_WIDTH = 100
BUTTON_HEIGHT = 30
PADDING = 10


def load_images():
    """
    Load images for pygame
    """
    pieces = ["wp", "wR", "wN", "wB", "wQ", "wK", "bp", "bR", "bN", "bB", "bQ", "bK"]

    images_folder = os.path.join(os.getcwd(), 'images')
    for piece in pieces:
        image_path = os.path.join(images_folder, piece + '.png')
        IMAGES[piece] = p.transform.scale(p.image.load(image_path), (SQ_SIZE, SQ_SIZE))


def main():
    p.init()
    screen = p.display.set_mode((WIDTH + MOVE_LOG_PANEL_WIDTH, HEIGHT))
    p.display.set_caption("Chess Game with GUI")
    clock = p.time.Clock()

    # pygame_gui Manager
    manager = pygame_gui.UIManager((WIDTH + MOVE_LOG_PANEL_WIDTH, HEIGHT))

    # Create buttons
    undo_button = pygame_gui.elements.UIButton(
        relative_rect=p.Rect((WIDTH + PADDING, PADDING), (BUTTON_WIDTH, BUTTON_HEIGHT)),
        text='Undo',
        manager=manager
    )
    reset_button = pygame_gui.elements.UIButton(
        relative_rect=p.Rect((WIDTH + PADDING, PADDING*2 + BUTTON_HEIGHT), (BUTTON_WIDTH, BUTTON_HEIGHT)),
        text='Reset',
        manager=manager
    )
    white_button = pygame_gui.elements.UIButton(
        relative_rect=p.Rect((WIDTH + PADDING, PADDING*3 + BUTTON_HEIGHT*2), (BUTTON_WIDTH, BUTTON_HEIGHT)),
        text='Play White',
        manager=manager
    )
    black_button = pygame_gui.elements.UIButton(
        relative_rect=p.Rect((WIDTH + PADDING, PADDING*4 + BUTTON_HEIGHT*3), (BUTTON_WIDTH, BUTTON_HEIGHT)),
        text='Play Black',
        manager=manager
    )
    
    human_human_button = pygame_gui.elements.UIButton(
        relative_rect=p.Rect((WIDTH + PADDING, PADDING*5 + BUTTON_HEIGHT*4), (BUTTON_WIDTH, BUTTON_HEIGHT)),
        text='Human Human',
        manager=manager
    )
    ava_button = pygame_gui.elements.UIButton(
    relative_rect=p.Rect((WIDTH + PADDING, PADDING*6 + BUTTON_HEIGHT*5), (BUTTON_WIDTH + 40, BUTTON_HEIGHT)), # Điều chỉnh kích thước nếu cần
    text='Minimax vs Random',
    manager=manager
)

    # Initial game state
    gs = GameState()
    load_images()
    valid_moves = gs.get_valid_moves()
    move_made = False
    animate = False
    game_over = False
    player_one = True
    player_two = True
    sq_selected = ()
    player_clicks = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')
    agent = DQNAgent(device=device, loading=True)
    # agent.load_model("/home/lapquang/code/Agent/chess-engine/chess-game-AI-main/chess_dqn_model.pth")  # Load the trained model


    def get_human_turn() -> bool:
        return (gs.white_to_move and player_one) or (not gs.white_to_move and player_two)
    

    # print("all valid moves:{} {}".format(len(valid_moves),valid_moves))
    running = True
    while running:
        human_turn = get_human_turn()
        time_delta = clock.tick(MAX_FPS) / 1000.0
        for event in p.event.get():
            # Let GUI Manager process first
            manager.process_events(event)

            if event.type == p.QUIT:
                running = False

            # Pygame mouse events for move selection
            elif event.type == p.MOUSEBUTTONDOWN:
                location = p.mouse.get_pos()
                col = location[0]//SQ_SIZE
                row = location[1]//SQ_SIZE
                if sq_selected == (row, col) or col >= 8:
                    sq_selected = ()
                    player_clicks = []
                else:
                    sq_selected = (row, col)
                    player_clicks.append(sq_selected)
                if len(player_clicks) == 2:
                    move : Move = Move(player_clicks[0], player_clicks[1], gs.board)
                    if move in valid_moves:
                        move = valid_moves[valid_moves.index(move)]
                        gs.make_move(move)
                        move_made = True
                        animate = True
                        sq_selected = ()
                        player_clicks = []
                        print(move.get_chess_notation())
                    else:
                        player_clicks = [sq_selected]

            # GUI button events
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == undo_button:
                    if gs.moves_log:
                        gs.undo_move()
                    move_made = True
                    animate = False
                    player_one = True
                    player_two = True
                    human_turn = get_human_turn()
                elif event.ui_element == reset_button:
                    gs = GameState()
                    valid_moves = gs.get_valid_moves()
                    move_made = False
                    animate = False
                    game_over = False
                    player_one = True
                    player_two = True
                    human_turn = get_human_turn()
                elif event.ui_element == white_button:
                    player_one = True
                    player_two = False
                    human_turn = get_human_turn()
                elif event.ui_element == black_button:
                    player_one = False
                    player_two = True
                    human_turn = get_human_turn()
                elif event.ui_element == human_human_button:
                    player_one = True
                    player_two = True
                    human_turn = get_human_turn()
                elif event.ui_element == ava_button:
                    player_one = False
                    player_two = False
                    gs = GameState()
                    valid_moves = gs.get_valid_moves()
                    move_made = False
                    animate = False
                    game_over = False
                    sq_selected = ()
                    player_clicks = []
                    print("Selected Mode: Minimax (White) vs Random (Black)")
                    human_turn = get_human_turn() 
                                   
            

        # AI move
        # human_turn = (gs.white_to_move and player_one) or (not gs.white_to_move and player_two)
        # print("human turn: {}".format(human_turn))

       # if not game_over and not human_turn:
     #       AIMove = algorithm_utils.find_best_move_minimax(gs, valid_moves)
      #      if AIMove is None:
       #         AIMove = algorithm_utils.find_random_move(valid_moves)
       #     gs.make_move(AIMove)
       #     move_made = True
       #     animate = True

        # Tính toán xem có phải lượt AI không
        is_ai_turn = (gs.white_to_move and not player_one) or \
                     (not gs.white_to_move and not player_two)

        if not game_over and is_ai_turn: # Bắt đầu khối lệnh xử lý AI
            # --- TOÀN BỘ CODE BÊN DƯỚI PHẢI THỤT VÀO 1 CẤP SO VỚI DÒNG IF NÀY ---
            AIMove = None # Khởi tạo AIMove là None

            # LẤY VALID MOVES NGAY ĐẦU KHỐI LỆNH AI
            current_valid_moves = gs.get_valid_moves()

            # Kiểm tra xem có nước đi hợp lệ không
            if not current_valid_moves:
                print("No valid moves for AI.") # Không còn nước đi hợp lệ
            else:
                # Nếu có nước đi hợp lệ, tiếp tục logic chọn AI và tìm nước đi
                if player_one == False and player_two == False: # Chế độ AvA
                    if gs.white_to_move: # Lượt của Minimax (Trắng)
                        print("Thinking: Minimax (White)...")
                        # SỬA BIẾN: Dùng current_valid_moves
                        # AIMove = algorithm_utils.find_best_move_minimax(gs, current_valid_moves)
                        AIMove = agent.inference(gs, current_valid_moves)
                    else: # Lượt của Random (Đen)
                        print("Thinking: Random (Black)...")
                        # AIMove = agent.inference(gs, current_valid_moves)
                        # AIMove = algorithm_utils.find_best_move_minimax(gs, current_valid_moves)
                        AIMove = algorithm_utils.find_random_move(current_valid_moves)
                elif not player_one and gs.white_to_move: # Chế độ PvA, AI là Trắng (Minimax)
                     print("Thinking: AI Minimax (White)...")
                     # SỬA BIẾN: Dùng current_valid_moves
                     AIMove = algorithm_utils.find_best_move_minimax(gs, current_valid_moves)
                elif not player_two and not gs.white_to_move: # Chế độ PvA, AI là Đen (Minimax)
                     print("Thinking: AI Minimax (Black)...")
                     # SỬA BIẾN: Dùng current_valid_moves
                     AIMove = algorithm_utils.find_best_move_minimax(gs, current_valid_moves)

                # Xử lý nếu Minimax không tìm được nước (dùng Random thay thế)
                if AIMove is None and ((gs.white_to_move and not player_one) or (not gs.white_to_move and not player_two)):
                     is_minimax_turn = (player_one == False and player_two == False and gs.white_to_move) or \
                                       (not player_one and gs.white_to_move) or \
                                       (not player_two and not gs.white_to_move and player_one==False)
                     if is_minimax_turn:
                        print("AI Minimax failed to find a move, choosing random.")
                        AIMove = algorithm_utils.find_random_move(current_valid_moves)


            # Thực hiện nước đi nếu tìm được một nước đi (AIMove khác None)
            # Khối if/else này cũng phải nằm BÊN TRONG 'if not game_over and is_ai_turn:'
            if AIMove:
                gs.make_move(AIMove)
                move_made = True
                animate = True
                print(f"AI moved: {AIMove.get_chess_notation()}")
            # else: # Có thể bỏ else này nếu không muốn in gì khi AI không đi được
            #    if current_valid_moves: # Chỉ in lỗi nếu có nước đi mà AI vẫn ko chọn đc
            #        print("AI failed to select a valid move.")

        # --- KẾT THÚC KHỐI LỆNH XỬ LÝ AI ---

        # end phan code em chinh sua 
            
        # After move handling
        if move_made:
            if animate:
                animateMove(gs.moves_log[-1], screen, gs.board, clock)
            valid_moves = gs.get_valid_moves()
            move_made = False
            animate = False

        # Draw board and pieces
        draw_game_state(screen, gs, valid_moves, sq_selected)

        # Draw GUI
        manager.update(time_delta)
        manager.draw_ui(screen)

        # Check end game
        if gs.check_mate or gs.stale_mate:
            game_over = True
            if gs.stale_mate:
                game_over = True
                drawEndGameText(screen, "DRAW")
            else:
                
                text_to_draw = "{} WIN".format("WHITE" if gs.white_to_move else "WHITE")
                drawEndGameText(screen, text_to_draw)
                
        p.display.flip()


def draw_board(screen):
    colors = [p.Color("white"), p.Color("grey")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            p.draw.rect(screen, color, p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))


def draw_pieces(screen, board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))


def draw_game_state(screen, gs, valid_moves, sq_selected):
    draw_board(screen)
    highlight_move(screen, gs, valid_moves, sq_selected)
    draw_pieces(screen, gs.board)
    draw_moveslog(screen, gs)

def highlight_move(screen, gs: GameState, validMoves: list[Move], sqSelected):
    sq = p.Surface((SQ_SIZE, SQ_SIZE))
    sq.set_alpha(100)
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.white_to_move else 'b'): #sqSelected is a piece that can be moved
            #highlight selected square
            sq.fill(p.Color("blue"))
            screen.blit(sq, (c * SQ_SIZE, r * SQ_SIZE))
            #highlight validmoves
            sq.fill(p.Color("cyan"))
            for move in validMoves:
                if move.start_row == r and move.start_col == c:
                    screen.blit(sq, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))

    if gs.in_check:
        if gs.white_to_move:
            sq.fill(p.Color("red"))
            screen.blit(sq, (gs.white_king_loc[1] * SQ_SIZE, gs.white_king_loc[0] * SQ_SIZE))
        else:
            sq.fill(p.Color("red"))
            screen.blit(sq, (gs.black_king_loc[1] * SQ_SIZE, gs.black_king_loc[0] * SQ_SIZE))
    
    if len(gs.moves_log) != 0:
        sq.fill(p.Color("yellow"))
        screen.blit(sq, (gs.moves_log[-1].start_col * SQ_SIZE, gs.moves_log[-1].start_row * SQ_SIZE))
        screen.blit(sq, (gs.moves_log[-1].end_col * SQ_SIZE, gs.moves_log[-1].end_row * SQ_SIZE))


def animateMove(move: Move, screen, board, clock):
    colors = [p.Color("white"), p.Color("grey")]
    dR = move.end_row - move.start_row
    dC = move.end_col - move.start_col
    sqDistance = math.sqrt(abs(move.end_row - move.start_row)*abs(move.end_row - move.start_row) +
                           abs(move.end_col - move.start_col)*abs(move.end_col - move.start_col))
    sqDistance = int(sqDistance)
    framesPerSquare = 12 // sqDistance
    frameCount = (abs(dR) + abs(dC)) * framesPerSquare
    for frame in range(frameCount + 1):
        r, c = (move.start_row + dR*frame/frameCount, move.start_col + dC*frame/frameCount)
        draw_board(screen)
        draw_pieces(screen, board)
        color = colors[(move.end_row + move.end_col) % 2]
        endSquare = p.Rect(move.end_col*SQ_SIZE, move.end_row*SQ_SIZE, SQ_SIZE, SQ_SIZE)
        p.draw.rect(screen, color, endSquare)
        if move.piece_captured != "--":
            if move.is_enpassant_move:
                enPassantRow = (move.end_row + 1) if move.piece_captured[0] == 'b' else (move.end_row - 1)
                endSquare = p.Rect(move.end_col*SQ_SIZE, enPassantRow*SQ_SIZE, SQ_SIZE, SQ_SIZE)
            screen.blit(IMAGES[move.piece_captured], endSquare)
        if move.piece_move != "--":
            screen.blit(IMAGES[move.piece_move], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
        p.display.flip()
        clock.tick(144)

def draw_game_state(screen, gs: GameState, validMoves, sqSelected):
    draw_board(screen)
    highlight_move(screen, gs, validMoves, sqSelected)
    draw_pieces(screen, gs.board)
    draw_moveslog(screen, gs)

def draw_board(screen):
    colors = [p.Color("white"), p.Color("grey")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r + c) % 2)]
            p.draw.rect(screen, color, p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def draw_pieces(screen, board):
    for row in range(DIMENSION):
        for col in range(DIMENSION):
            piece = board[row][col]
            if piece != "--":
                screen.blit(IMAGES[piece], p.Rect(col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

def drawEndGameText(screen, text):
    font = p.font.SysFont("Verdana", 32, True, False)
    textObject = font.render(text, False, p.Color("black"))
    textLocation = p.Rect(0, 0, WIDTH, HEIGHT).move(WIDTH/2 - textObject.get_width()/2, HEIGHT/2 - textObject.get_height()/2)
    screen.blit(textObject, textLocation)
    textObject = font.render(text, False, p.Color("red"))
    screen.blit(textObject, textLocation.move(2, 2))

def draw_moveslog(screen, gs: GameState):
    return 
    moves_logRect = p.Rect(WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, p.Color("black"), moves_logRect)
    moves_log = gs.moves_log
    moveTexts = []
    for i in range(0, len(moves_log), 2):
        moveString = str(i//2 + 1) + ". " + str(moves_log[i]) + " "
        if i+1 < len(moves_log):
            moveString += str(moves_log[i+1]) + "   "
        moveTexts.append(moveString)
    
    padding = 5
    movesPerRow = 3
    lineSpacing = 2
    textY = padding
    for i in range(0, len(moveTexts), movesPerRow):
        text = ""
        font = p.font.SysFont("Verdana", 13, True, False)
        for j in range(movesPerRow):
            if i+j < len(moveTexts):
                text += moveTexts[i+j]
        textObject = font.render(text, False, p.Color("white"))
        textLocation = moves_logRect.move(padding, textY)
        screen.blit(textObject, textLocation)
        textY += textObject.get_height() + lineSpacing

if __name__ == "__main__":
    main()
