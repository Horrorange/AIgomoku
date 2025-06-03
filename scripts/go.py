import pygame
import sys
from pygame.locals import QUIT, KEYDOWN, MOUSEBUTTONDOWN, K_ESCAPE # 添加 MOUSEBUTTONDOWN 和 K_ESCAPE

# Initialize the pygame module
pygame.init()

# --- Constants ---
SCREEN_WIDTH = 670
SCREEN_HEIGHT = 670
BOARD_OFFSET = 27
CELL_SIZE = 44
HALF_CELL = CELL_SIZE // 2
GRID_LINES = 15 # Number of lines, so 14x14 cells, or 15x15 intersections
BOARD_END_PX = BOARD_OFFSET + (GRID_LINES - 1) * CELL_SIZE

SCREEN_COLOR = [238, 154, 73]
LINE_COLOR = [0, 0, 0]
WHITE_COLOR = [255, 255, 255]
BLACK_COLOR = [0, 0, 0]
HIGHLIGHT_COLOR = [0, 229, 238]
CURSOR_COLOR = [100, 100, 100] # For mouse cursor highlight when no valid spot

# Set the windows
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Gomoku") # Set window title

# --- Game State ---
# board_state[row][col] = 0 (empty), 1 (black), 2 (white)
# This is a more common and useful representation for game logic
board_state = [[0 for _ in range(GRID_LINES)] for _ in range(GRID_LINES)]
over_pos_visual = [] # For drawing: [[[x,y], color], ...] - for visual representation
current_player_color = BLACK_COLOR # Black goes first
player_turn = 1 # 1 for black, 2 for white
game_over = False

def screen_to_grid(screen_x, screen_y):
    '''Converts screen pixel coordinates to grid indices (row, col)'''
    if screen_x < BOARD_OFFSET - HALF_CELL or \
       screen_x > BOARD_END_PX + HALF_CELL or \
       screen_y < BOARD_OFFSET - HALF_CELL or \
       screen_y > BOARD_END_PX + HALF_CELL:
        return None, None # Outside playable area

    # Find the closest intersection
    grid_x = round((screen_x - BOARD_OFFSET) / CELL_SIZE)
    grid_y = round((screen_y - BOARD_OFFSET) / CELL_SIZE)

    if 0 <= grid_x < GRID_LINES and 0 <= grid_y < GRID_LINES:
        return grid_x, grid_y
    return None, None

def grid_to_screen(grid_x, grid_y):
    '''Converts grid indices to screen pixel coordinates for the center of the intersection'''
    screen_x = BOARD_OFFSET + grid_x * CELL_SIZE
    screen_y = BOARD_OFFSET + grid_y * CELL_SIZE
    return screen_x, screen_y

def is_valid_move(grid_x, grid_y):
    '''Checks if the move is on the board and the spot is empty'''
    if grid_x is None or grid_y is None:
        return False
    return 0 <= grid_x < GRID_LINES and \
           0 <= grid_y < GRID_LINES and \
           board_state[grid_y][grid_x] == 0 # board_state is [row][col] so [grid_y][grid_x]

# --- Placeholder for win checking ---
def check_win(player, last_move_gx, last_move_gy):
    """
    Checks if the current player has won.
    :param player: 1 for black, 2 for white
    :param last_move_gx: grid x of the last move
    :param last_move_gy: grid y of the last move
    :return: True if player won, False otherwise
    """
    if last_move_gx is None or last_move_gy is None: # Should not happen if called after a valid move
        return False

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Horizontal, Vertical, Diagonal \, Diagonal /
    for dr, dc in directions:
        count = 1 # Count the last placed stone
        # Check in one direction
        for i in range(1, 5):
            r, c = last_move_gy + dr * i, last_move_gx + dc * i
            if 0 <= r < GRID_LINES and 0 <= c < GRID_LINES and board_state[r][c] == player:
                count += 1
            else:
                break
        # Check in the opposite direction
        for i in range(1, 5):
            r, c = last_move_gy - dr * i, last_move_gx - dc * i
            if 0 <= r < GRID_LINES and 0 <= c < GRID_LINES and board_state[r][c] == player:
                count += 1
            else:
                break
        if count >= 5:
            return True
    return False

def is_board_full():
    for r in range(GRID_LINES):
        for c in range(GRID_LINES):
            if board_state[r][c] == 0:
                return False
    return True


# --- Main Loop ---
while True:
    mouse_gx, mouse_gy = None, None # Grid coordinates of mouse hover
    valid_hover_pos_screen = None # Screen coordinates for valid hover highlight

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_r and game_over: # Press R to restart
                board_state = [[0 for _ in range(GRID_LINES)] for _ in range(GRID_LINES)]
                over_pos_visual = []
                current_player_color = BLACK_COLOR
                player_turn = 1
                game_over = False


        if not game_over:
            mouse_screen_x, mouse_screen_y = pygame.mouse.get_pos()
            # Convert mouse position to grid indices
            temp_gx, temp_gy = screen_to_grid(mouse_screen_x, mouse_screen_y)

            if temp_gx is not None and temp_gy is not None: # If mouse is on a valid grid intersection
                mouse_gx, mouse_gy = temp_gx, temp_gy
                if is_valid_move(mouse_gx, mouse_gy):
                    valid_hover_pos_screen = grid_to_screen(mouse_gx, mouse_gy)

            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse button
                    if valid_hover_pos_screen: # If clicked on a valid empty spot
                        # Place piece
                        board_state[mouse_gy][mouse_gx] = player_turn # Update logical board
                        screen_x_place, screen_y_place = grid_to_screen(mouse_gx, mouse_gy)
                        over_pos_visual.append([[screen_x_place, screen_y_place], current_player_color])

                        # Check for win
                        if check_win(player_turn, mouse_gx, mouse_gy):
                            print(f"{'Black' if player_turn == 1 else 'White'} wins!")
                            game_over = True
                        elif is_board_full():
                            print("It's a draw!")
                            game_over = True
                        else:
                            # Switch player
                            if player_turn == 1:
                                player_turn = 2
                                current_player_color = WHITE_COLOR
                            else:
                                player_turn = 1
                                current_player_color = BLACK_COLOR

    # --- Drawing ---
    screen.fill(SCREEN_COLOR)

    # Draw the grid lines and center dots
    # Star points (tianyuan and other key points)
    star_points_grid = [
        (3, 3), (11, 3), (3, 11), (11, 11),  # Corners
        (7, 7) # Center (tianyuan)
    ]
    if GRID_LINES == 19: # For 19x19 Go board, more star points
        star_points_grid.extend([(3,9), (9,3), (15,3), (3,15), (15,9), (9,15), (15,15), (9,9)])


    for i in range(GRID_LINES):
        x_coord = BOARD_OFFSET + i * CELL_SIZE
        y_coord = BOARD_OFFSET + i * CELL_SIZE

        line_width = 4 if i == 0 or i == GRID_LINES - 1 else 2
        # Horizontal lines
        pygame.draw.line(screen, LINE_COLOR, [BOARD_OFFSET, y_coord], [BOARD_END_PX, y_coord], line_width)
        # Vertical lines
        pygame.draw.line(screen, LINE_COLOR, [x_coord, BOARD_OFFSET], [x_coord, BOARD_END_PX], line_width)

    for gx, gy in star_points_grid:
        if 0 <= gx < GRID_LINES and 0 <= gy < GRID_LINES:
            sp_x, sp_y = grid_to_screen(gx, gy)
            pygame.draw.circle(screen, LINE_COLOR, [sp_x, sp_y], 6, 0)


    # Highlight potential move
    if valid_hover_pos_screen and not game_over:
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, [valid_hover_pos_screen[0]-HALF_CELL, valid_hover_pos_screen[1]-HALF_CELL, CELL_SIZE, CELL_SIZE], 2, 1)
    elif mouse_gx is not None and mouse_gy is not None and not game_over: # If hovering over a valid intersection (even if occupied or invalid)
        temp_screen_x, temp_screen_y = grid_to_screen(mouse_gx, mouse_gy)
        pygame.draw.rect(screen, CURSOR_COLOR, [temp_screen_x-HALF_CELL, temp_screen_y-HALF_CELL, CELL_SIZE, CELL_SIZE], 1, 1)


    # Draw the placed chess pieces
    for val in over_pos_visual:
        pygame.draw.circle(screen, val[1], val[0], HALF_CELL - 2 , 0) # piece_radius slightly smaller than half_cell

    # Display game over message
    if game_over:
        font = pygame.font.Font(None, 74)
        text_content = ""
        if check_win(1, None, None): # A bit hacky, better to store winner
             # Re-check winner if needed or store winner state
            winner_found = False
            for r_idx, r_val in enumerate(board_state):
                for c_idx, c_val in enumerate(r_val):
                    if c_val != 0 and check_win(c_val, c_idx, r_idx):
                        text_content = f"{'Black' if c_val == 1 else 'White'} wins!"
                        winner_found = True
                        break
                if winner_found: break
            if not text_content and is_board_full(): # if no winner and board is full
                 text_content = "It's a Draw!"

        elif is_board_full(): # Check again, in case check_win was based on last move
            text_content = "It's a Draw!"
        else: # One player must have won
             # This logic needs to be robust, find out who won
            if any(check_win(1, c, r) for r in range(GRID_LINES) for c in range(GRID_LINES) if board_state[r][c] == 1):
                 text_content = "Black wins!"
            elif any(check_win(2, c, r) for r in range(GRID_LINES) for c in range(GRID_LINES) if board_state[r][c] == 2):
                 text_content = "White wins!"


        if text_content:
            text = font.render(text_content, True, (200, 0, 0))
            text_rect = text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 - 50))
            screen.blit(text, text_rect)

        restart_font = pygame.font.Font(None, 40)
        restart_text = restart_font.render("Press 'R' to Restart", True, (0,100,0))
        restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 20))
        screen.blit(restart_text, restart_rect)


    pygame.display.update()