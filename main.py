import math
import pygame as p # type: ignore
import chess_engine 
from chess_engine import Move, GameState
import algorithm_utils
import os
# from GUI import ChessGameGUI 

WIDTH = HEIGHT = 512
MOVE_LOG_PANEL_WIDTH = 290
MOVE_LOG_PANEL_HEIGHT = HEIGHT
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 240
IMAGES = {}

def load_images():
    """
    Load images for pygame
    """
    pieces = ["wp", "wR", "wN", "wB", "wQ", "wK", "bp", "bR", "bN", "bB", "bQ", "bK"]
    images_folder = os.path.join(os.getcwd(), 'images')  # Sử dụng os.path để tạo đường dẫn
    for piece in pieces:
        image_path = os.path.join(images_folder, piece + '.png')
        IMAGES[piece] = p.transform.scale(p.image.load(image_path), (SQ_SIZE, SQ_SIZE))

def start_game(mode):
    gs = GameState()  # Khởi tạo GameState
    load_images()  # Tải các hình ảnh quân cờ

    if mode == "Người vs Người":
        # Logic cho chế độ Người vs Người
        pass
    elif mode == "Người vs AI":
        # Logic cho chế độ Người vs AI
        pass
    elif mode == "AI vs AI":
        # Logic cho chế độ AI vs AI
        pass

    return gs  # Trả về trạng thái game sau khi bắt đầu

def undo_move(gs):
    if gs.moves_log:
        gs.undo_move()  # Gọi phương thức undo_move() trong GameState
    return gs

def reset_game():
    gs = GameState()  # Khởi tạo lại GameState
    return gs

def play_white(gs):
    # Logic cho chế độ Người vs AI khi người chơi chọn chơi bên trắng
    pass

def play_black(gs):
    # Logic cho chế độ Người vs AI khi người chơi chọn chơi bên đen
    pass

def main():
    p.init()
    screen = p.display.set_mode((WIDTH + MOVE_LOG_PANEL_WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = chess_engine.GameState()
    valid_moves = gs.get_valid_moves()
    move_made = False
    animate = False
    game_over = False
    player_one = True   
    player_two = False
    load_images()
    sq_selected = ()
    player_clicks = []

    running = True
    while running:
        humanTurn = (gs.white_to_move and player_one) or (not gs.white_to_move and player_two)
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif e.type == p.MOUSEBUTTONDOWN:
                if not game_over and humanTurn:
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
            elif e.type == p.KEYDOWN:
                if e.key == p.K_z:
                    gs.undo_move()
                    move_made = True
                    animate = False
                    game_over = False
                    player_one = True
                    player_two = True
                elif e.key == p.K_r:
                    gs = chess_engine.GameState()
                    valid_moves = gs.get_valid_moves()
                    move_made = False
                    animate = False
                    game_over = False
                    player_one = True
                    player_two = True
                    sq_selected = ()
                    player_clicks = []
                elif e.key == p.K_q:
                    player_one = False
                    player_two = True
                elif e.key == p.K_e:
                    player_one = True
                    player_two = False

        if move_made:
            if animate:
                animateMove(gs.moves_log[-1], screen, gs.board, clock)
            valid_moves = gs.get_valid_moves()
            move_made = False
            animate = False

        ''' AI move finder '''
        if not game_over and not humanTurn:
            AIMove = algorithm_utils.find_best_move_minimax(gs, valid_moves)
            if AIMove is None:   #when begin the game
                AIMove = algorithm_utils.find_random_move(valid_moves)
            gs.make_move(AIMove)
            move_made = True
            animate = True
            print(AIMove.get_chess_notation())


        draw_game_state(screen, gs, valid_moves, sq_selected)

        if gs.check_mate or gs.stale_mate:
            game_over = True
            if gs.stale_mate:
                game_over = True
                drawEndGameText(screen, "DRAW")
            else:
                
                text_to_draw = "{} WIN".format("WHITE" if gs.white_to_move else "WHITE")
                drawEndGameText(screen, text_to_draw)

        clock.tick(MAX_FPS)
        p.display.flip()

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
    
    
################## 
# 1) Tao them tkiner -> text, button -> chon agent
# 2) Co 1 che do: ben trang la agent cua minh, ben den la agent random
# 3) Che do nguoi vs agent: 
###################
# BTL 2)
# a. Tao tinker co text va button
# b. 3 che do: nguoi vs nguoi, nguoi vs agent, agent minimax vs agent random
# c. Evaluation agent: simulate 1000 ban co -> danh gia khar nang thang thua cua moi agent