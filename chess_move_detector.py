import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import tkinter as tk
from tkinter import filedialog

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Match result
match_result = []

# Erosion kernel
EROSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Global variables for manual point selection
selected_points = []
click_count = 0

def select_video_file():
    """Open a file dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.mkv *.avi")]
    )
    return file_path

def get_video_duration(video_path):
    """Get the duration of a video in seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        raise ValueError("Cannot open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps <= 0:
        logging.error("Invalid FPS value in video")
        raise ValueError("Invalid FPS value")
    
    duration = frame_count / fps
    logging.info(f"Video duration: {duration:.2f} seconds")
    return duration

def extract_frames_in_duration(video_path, start_time, end_time, frame_interval=1.0):
    """Extract frames from a video within a specified duration at given intervals."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        raise ValueError("Cannot open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logging.error("Invalid FPS value in video")
        raise ValueError("Invalid FPS value")
    
    duration = get_video_duration(video_path)
    if start_time < 0 or end_time > duration or start_time >= end_time:
        logging.error(f"Invalid duration: start_time={start_time}s, end_time={end_time}s, video_duration={duration}s")
        raise ValueError(f"Invalid duration: start_time={start_time}s, end_time={end_time}s, video_duration={duration}s")
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_step = max(1, int(frame_interval * fps))
    
    frames = []
    frame_times = []
    
    for frame_num in range(start_frame, end_frame + 1, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to extract frame {frame_num} from video")
            continue
        
        frame_time = frame_num / fps
        logging.info(f"Extracted frame at {frame_time:.2f} seconds (frame {frame_num})")
        frames.append(frame)
        frame_times.append(frame_time)
    
    cap.release()
    
    if not frames:
        logging.error(f"No frames extracted in the specified duration: {start_time}s to {end_time}s")
        raise ValueError("No frames extracted")
    
    return frames, frame_times, fps

def extract_digital_board(image, debug=False):
    """Detect chessboard region from image."""
    print('Detecting first chessboard square from image...')
    h, w = image.shape[:2]
    crop = image[0:h, 0:w]

    if crop.size == 0:
        raise ValueError("Cropped image is empty")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closed, 50, 150)

    contour_modes = [cv2.RETR_EXTERNAL, cv2.RETR_TREE]
    best_quad = None

    if debug:
        os.makedirs('./debug_frames', exist_ok=True)
        cv2.imwrite('./debug_frames/_1_gray.png', gray)
        cv2.imwrite('./debug_frames/_2_blurred.png', blurred)
        cv2.imwrite('./debug_frames/_3_thresh.png', thresh)
        cv2.imwrite('./debug_frames/_4_closed.png', closed)
        cv2.imwrite('./debug_frames/_5_edges.png', edges)

    for mode in contour_modes:
        contours, _ = cv2.findContours(edges.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)

        if debug:
            debug_img = crop.copy()
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(f'./debug_frames/_contours_mode_{mode}.png', debug_img)

        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(approx) == 4 and cv2.contourArea(approx) > 5000 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)
                rect = order_points(pts)

                side_lengths = [
                    np.linalg.norm(rect[0] - rect[1]),
                    np.linalg.norm(rect[1] - rect[2]),
                    np.linalg.norm(rect[2] - rect[3]),
                    np.linalg.norm(rect[3] - rect[0])
                ]

                if min(side_lengths) < 50:
                    continue

                aspect_ratio = max(side_lengths) / min(side_lengths)
                if 0.8 < aspect_ratio < 1.2:
                    best_quad = rect
                    break
        if best_quad is not None:
            break

    if best_quad is None:
        print("No valid square detected.")
        return crop, None

    square_w, square_h = 75, 75
    board_w, board_h = 8 * square_w, 8 * square_h

    top_left = best_quad[0]

    chessboard_corners = np.array([
        top_left,
        top_left + np.array([board_w, 0]),
        top_left + np.array([board_w, board_h]),
        top_left + np.array([0, board_h])
    ], dtype=np.float32)

    if debug:
        debug_img = crop.copy()
        cv2.polylines(debug_img, [best_quad.astype(int)], True, (255, 0, 0), 3)
        cv2.polylines(debug_img, [chessboard_corners.astype(int)], True, (0, 0, 255), 3)
        cv2.imwrite('./debug_frames/_detected_squares.png', debug_img)

    print(f"First square corners: {best_quad}")
    print(f"Computed chessboard corners: {chessboard_corners}")

    return crop, chessboard_corners

def order_points(pts):
    """Return consistent order: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def mouse_callback(event, x, y, flags, param):
    """Callback for mouse clicks to select chessboard corners."""
    global selected_points, click_count
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append([x, y])
        click_count += 1
        logging.info(f"Point {click_count} selected: ({x}, {y})")

def select_board_corners(frame):
    """Allow user to manually select four chessboard corners."""
    global selected_points, click_count
    selected_points = []
    click_count = 0
    
    window_name = "Select Chessboard Corners (Click 4 points, press 'q' to confirm)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while click_count < 4:
        display_frame = frame.copy()
        for i, pt in enumerate(selected_points):
            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"P{i+1}", (int(pt[0]) + 10, int(pt[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and click_count == 4:
            break
    
    cv2.destroyWindow(window_name)
    if len(selected_points) == 4:
        return np.array(selected_points, dtype="float32")
    else:
        logging.error("Exactly 4 points must be selected")
        raise ValueError("Exactly 4 points must be selected")

def blur_border(img, border_size=10, blur_kernel=(15, 15)):
    """Apply a blurred border to an image."""
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, blur_kernel, 0)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.rectangle(mask, (border_size, border_size), (w - border_size, h - border_size), 1, -1)
    mask = cv2.GaussianBlur(mask, (border_size * 2 + 1, border_size * 2 + 1), 0)

    if len(img.shape) == 2:
        result = img.astype(np.float32) * mask + blurred.astype(np.float32) * (1 - mask)
    else:
        result = img.astype(np.float32)
        for c in range(3):
            result[..., c] = result[..., c] * mask + blurred[..., c] * (1 - mask)

    return np.clip(result, 0, 255).astype(np.uint8)

def load_templates(template_dir="templates", debug_dir="debug_output", debug=True):
    """Load chess piece templates."""
    pieces = ["P", "P_g", "N", "N_g", "B", "B_g", "R", "R_g", "Q", "Q_g", "K", "K_g", 
              "pb", "pb_g", "nb", "nb_g", "bb", "bb_g", "rb", "rb_g", "qb", "qb_g", "kb", "kb_g"]
    templates = {}
    os.makedirs(debug_dir, exist_ok=True)

    for p in pieces:
        path = os.path.join(template_dir, f"{p}.png")  # Corrected file extension
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.error(f"Template for {p} not found at {path}")
            raise ValueError(f"Template for {p} not found at {path}")

        resized = cv2.resize(img, (80, 80))
        eroded = cv2.erode(resized, EROSION_KERNEL, iterations=1)
        blurred = blur_border(eroded, border_size=6)
        templates[p] = blurred

        if debug:
            cv2.imwrite(os.path.join(debug_dir, f"{p}_eroded.png"), eroded)

    return templates

def match_piece(square_img, img_name, templates, frame_idx, threshold=0.6, debug=False):
    """Match a chess piece in a square image using template matching."""
    if square_img.size == 0 or square_img.shape[0] == 0 or square_img.shape[1] == 0:
        logging.warning(f"Empty square image: {img_name}")
        return None

    square_gray = square_img if len(square_img.shape) == 2 else cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    square_resized = cv2.resize(square_gray, (80, 80))
    square_resized = cv2.erode(square_resized, EROSION_KERNEL, iterations=1)
    square_resized = blur_border(square_resized, border_size=6)

    max_val = 0
    best_match = None
    debug_frames = []

    for piece, template in templates.items():
        try:
            res = cv2.matchTemplate(square_resized, template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if debug:
                vis = np.hstack([square_resized, template])
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                cv2.putText(vis_bgr, f"Piece: {piece} Score: {val:.2f}", (5, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                debug_frames.append((piece, vis_bgr))
            if val > max_val:
                max_val = val
                best_match = piece
        except cv2.error as e:
            logging.error(f"matchTemplate error for {piece} in {img_name}: {e}")
            continue


    if max_val >= threshold:
        if False:
            frame_debug_dir = f'./debug_frames/frame_{frame_idx:03d}/match'
            os.makedirs(frame_debug_dir, exist_ok=True)
            best_vis = np.hstack([square_resized, templates[best_match]])
            best_vis_bgr = cv2.cvtColor(best_vis, cv2.COLOR_GRAY2BGR)
            cv2.putText(best_vis_bgr, f"{best_match} ({max_val:.2f})", (5, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(f'{frame_debug_dir}/{img_name}_best_match.png', best_vis_bgr)
            match_result.append(f"{img_name} match = {best_match}") 
        return best_match
    return None

def warp_board(crop, points):
    """Warp the chessboard region to a standard 552x552 image."""
    rect = order_points(points)
    dst = np.array([
        [0, 0],
        [551, 0],
        [551, 551],
        [0, 551]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(crop, M, (552, 552))

    if not warped.size:
        raise ValueError("Warped image is empty or invalid")

    return warped, M

def split_into_squares(board_img, debug_dir="./debug_frames"):
    """Split the warped board into 8x8 squares and save debug images."""
    os.makedirs(debug_dir, exist_ok=True)
    squares = []
    square_names = []
    square_positions = []
    height, width = board_img.shape[:2]
    dy, dx = 69, 69  # To be calibrated or algorithm to detect them 

    if height < 8 * dy or width < 8 * dx:
        logging.warning(f"Warped board too small: {width}x{height}, need at least {8*dx}x{8*dy}")
        top_pad = max(0, 8 * dy - height)
        left_pad = max(0, 8 * dx - width)
        board_img = cv2.copyMakeBorder(board_img, 0, top_pad, 0, left_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        logging.info(f"Padded board to {board_img.shape[1]}x{board_img.shape[0]}")

    for row in range(8):
        for col in range(8):
            y_start, y_end = row * dy, (row + 1) * dy
            x_start, x_end = col * dx, (col + 1) * dx
            square = board_img[y_start:y_end, x_start:x_end]
            if not square.size or square.shape[0] < dy or square.shape[1] < dx:
                logging.warning(f"Empty or undersized square at row {row}, col {col}")
                continue
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if len(square.shape) == 3 else square
            resized = cv2.resize(gray, (80, 80))
            name = f'square_r{row}_c{col}.png'
            debug_path = os.path.join(debug_dir, name)
            #cv2.imwrite(debug_path, resized)
            squares.append(resized)
            square_names.append(name)
            square_positions.append((row, col))
    return squares, square_names, square_positions

def generate_fen(squares, square_names, templates, frame_idx, debug=False):
    """
    Generate FEN string from the board squares.

    Args:
        squares: List of square images (80x80 grayscale, preprocessed).
        square_names: List of square names (e.g., 'square_r0_c0.png').
        templates: Dictionary of preprocessed template images from load_templates.
        frame_idx: Index of the current frame for debugging.
        debug: Boolean to enable debug outputs.

    Returns:
        tuple: (FEN string, 8x8 board state list)
    """
    board = [['' for _ in range(8)] for _ in range(8)]
    fen_map = {
        'P': 'P', 'P_g': 'P', 'N': 'N', 'N_g': 'N', 'B': 'B', 'B_g': 'B',
        'R': 'R', 'R_g': 'R', 'Q': 'Q', 'Q_g': 'Q', 'K': 'K', 'K_g': 'K',
        'pb': 'p', 'pb_g': 'p', 'nb': 'n', 'nb_g': 'n', 'bb': 'b', 'bb_g': 'b',
        'rb': 'r', 'rb_g': 'r', 'qb': 'q', 'qb_g': 'q', 'kb': 'k', 'kb_g': 'k'
    }

    # Detect pieces in each square
    for i, (square, name) in enumerate(zip(squares, square_names)):
        row, col = i // 8, i % 8
        piece = match_piece(square, name, templates, frame_idx, threshold=0.6, debug=debug)
        if piece and piece in fen_map:
            board[row][col] = fen_map[piece]
            if debug:
                logging.debug(f"Square {name} (r{row}, c{col}): Detected {piece} -> {fen_map[piece]}")
        else:
            board[row][col] = ''
            if debug:
                logging.debug(f"Square {name} (r{row}, c{col}): No piece detected")

    # Validate piece counts to catch preprocessing errors
    piece_counts = {}
    for row in board:
        for piece in row:
            if piece:
                piece_counts[piece] = piece_counts.get(piece, 0) + 1

    # Check for plausible piece counts (e.g., max 8 pawns per side)
    max_counts = {'P': 8, 'p': 8, 'N': 2, 'n': 2, 'B': 2, 'b': 2,
                  'R': 2, 'r': 2, 'Q': 9, 'q': 9, 'K': 1, 'k': 1}  # Allowing promoted queens
    for piece, count in piece_counts.items():
        if count > max_counts.get(piece, 0):
            logging.warning(f"Invalid piece count for {piece}: {count} exceeds maximum {max_counts.get(piece, 0)}")
            if debug:
                frame_debug_dir = f'./debug_frames/frame_{frame_idx:03d}'
                os.makedirs(frame_debug_dir, exist_ok=True)
                with open(f'{frame_debug_dir}/piece_counts.txt', 'w') as f:
                    f.write(f"Invalid piece counts: {piece_counts}\n")
            return None, board  # Return None FEN to indicate error

    # Generate FEN string
    fen_rows = []
    for row in board:
        empty = 0
        fen_row = ''
        for square in row:
            if square == '':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += square
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    fen = '/'.join(fen_rows) + ' w KQkq - 0 1'
    if debug:
        logging.info(f"Generated FEN for frame {frame_idx}: {fen}")
        frame_debug_dir = f'./debug_frames/frame_{frame_idx:03d}'
        os.makedirs(frame_debug_dir, exist_ok=True)
        with open(f'{frame_debug_dir}/board_state.txt', 'w') as f:
            f.write(f"FEN: {fen}\n")
            for row in board:
                f.write(f"{row}\n")

    return fen, board

def detect_movement(prev_board, curr_board, prev_boards=None, window_size=3):
    """
    Detect chess piece movement between two board states with temporal context.
    
    Args:
        prev_board: Previous board state (8x8 list of piece symbols or '')
        curr_board: Current board state (8x8 list of piece symbols or '')
        prev_boards: List of previous board states for temporal context (optional)
        window_size: Number of previous frames to consider for move confirmation
    
    Returns:
        List of detected moves in algebraic notation (e.g., 'pe2-e4', 'Nxd4')
    """
    moves = []
    piece_counts = {}  # Track piece counts to detect captures or occlusions
    fen_map = {'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
               'pb': 'p', 'nb': 'n', 'bb': 'b', 'rb': 'r', 'qb': 'q', 'kb': 'k'}

    # Initialize piece counts for current board
    for row in curr_board:
        for piece in row:
            if piece:
                piece_counts[piece] = piece_counts.get(piece, 0) + 1

    # Compare with previous board
    for row in range(8):
        for col in range(8):
            prev_piece = prev_board[row][col]
            curr_piece = curr_board[row][col]
            if prev_piece != curr_piece:
                to_square = f"{chr(97 + col)}{8 - row}"
                # Case 1: Empty to piece (piece moved to this square)
                if prev_piece == '' and curr_piece != '':
                    # Find where the piece came from
                    for r in range(8):
                        for c in range(8):
                            if prev_board[r][c] == curr_piece and curr_board[r][c] == '':
                                from_square = f"{chr(97 + c)}{8 - r}"
                                move = f"{fen_map.get(curr_piece, curr_piece)}{from_square}-{to_square}"
                                if prev_boards and not confirm_move(prev_boards, curr_piece, from_square, to_square, window_size):
                                    continue
                                moves.append(move)
                                break
                # Case 2: Piece to empty (piece moved away, handled above)
                elif prev_piece != '' and curr_piece == '':
                    continue
                # Case 3: Piece to different piece (capture)
                elif prev_piece != '' and curr_piece != '':
                    from_square = None
                    # Look for the source of curr_piece
                    for r in range(8):
                        for c in range(8):
                            if prev_board[r][c] == curr_piece and curr_board[r][c] == '':
                                from_square = f"{chr(97 + c)}{8 - r}"
                                break
                    if from_square:
                        move = f"{fen_map.get(curr_piece, curr_piece)}{from_square}x{to_square}"
                        if prev_boards and not confirm_move(prev_boards, curr_piece, from_square, to_square, window_size):
                            continue
                        moves.append(move)
                    else:
                        logging.debug(f"Possible occlusion at {to_square}: {prev_piece} -> {curr_piece}")

    # Validate piece counts to detect potential errors
    prev_piece_counts = {}
    for row in prev_board:
        for piece in row:
            if piece:
                prev_piece_counts[piece] = prev_piece_counts.get(piece, 0) + 1
    
    for piece, count in piece_counts.items():
        prev_count = prev_piece_counts.get(piece, 0)
        if count > prev_count:
            logging.debug(f"Piece {piece} count increased ({prev_count} -> {count}), possible misdetection")
            moves = [m for m in moves if not m.startswith(fen_map.get(piece, piece))]

    return moves

def confirm_move(prev_boards, piece, from_square, to_square, window_size):
    """
    Confirm a move by checking consistency across previous board states.
    
    Args:
        prev_boards: List of previous board states
        piece: Piece symbol (e.g., 'P', 'pb')
        from_square: Starting square (e.g., 'e2')
        to_square: Destination square (e.g., 'e4')
        window_size: Number of frames to check
    
    Returns:
        bool: True if move is consistent across frames
    """
    if not prev_boards or len(prev_boards) < window_size:
        return True  # Not enough data, assume valid
    
    # Convert algebraic notation to board indices
    def square_to_coords(square):
        col = ord(square[0]) - ord('a')
        row = 8 - int(square[1])
        return row, col

    from_row, from_col = square_to_coords(from_square)
    to_row, to_col = square_to_coords(to_square)
    
    # Check if the move is consistent in recent frames
    consistent_count = 0
    for board in prev_boards[-window_size:]:
        if (board[from_row][from_col] == piece and 
            (board[to_row][to_col] != piece or board[to_row][to_col] == '')):
            consistent_count += 1
    
    return consistent_count >= window_size // 2

def annotate_frame(frame, moves, frame_time, points, M):
    """Annotate the frame with detected moves, frame time, and highlight move squares in green."""
    annotated = frame.copy()
    
    dst = np.array([
        [0, 0],
        [551, 0],
        [551, 551],
        [0, 551]
    ], dtype="float32")
    M_inv = cv2.getPerspectiveTransform(dst, order_points(points))
    
    # Draw chessboard outline
    for i in range(4):
        pt1 = tuple(points[i].astype(int))
        pt2 = tuple(points[(i + 1) % 4].astype(int))
        cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)

    # Highlight move squares
    square_size = 552 / 8  # Size of each square in warped board (552x552)
    for move in moves:
        move = move[1:] if move[0] in 'prnbqkPRNBQK' else move
        if '-' in move:
            from_square, to_square = move.split('-')
        elif 'x' in move:
            from_square, to_square = move.split('x')
        else:
            continue

        def square_to_coords(square):
            col = ord(square[0]) - ord('a')
            row = 8 - int(square[1])
            return row, col

        try:
            from_row, from_col = square_to_coords(from_square)
            to_row, to_col = square_to_coords(to_square)

            for row, col in [(from_row, from_col), (to_row, to_col)]:
                top_left = np.array([col * square_size, row * square_size], dtype=np.float32)
                bottom_right = np.array([(col + 1) * square_size, (row + 1) * square_size], dtype=np.float32)
                square_pts = np.array([
                    top_left,
                    [bottom_right[0], top_left[1]],
                    bottom_right,
                    [top_left[0], bottom_right[1]]
                ], dtype=np.float32).reshape(-1, 1, 2)

                square_pts_orig = cv2.perspectiveTransform(square_pts, M_inv)
                square_pts_orig = square_pts_orig.astype(int).reshape(-1, 2)
                cv2.polylines(annotated, [square_pts_orig], isClosed=True, color=(0, 255, 0), thickness=2)
        except (ValueError, IndexError) as e:
            logging.warning(f"Failed to parse move '{move}': {e}")
            continue

    # Draw text annotations
    y_offset = 50
    cv2.putText(annotated, f"Time: {frame_time:.2f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    for i, move in enumerate(moves):
        cv2.putText(annotated, f"Move: {move}", (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotated

def main(start_time=0, end_time=10, frame_interval=1.0):
    """Main function to process video and detect chess moves, pausing on movement."""
    out = None
    try:
        # GUI for video selection
        video_path = select_video_file()
        if not video_path:
            logging.error("No video file selected")
            raise ValueError("No video file selected")
        
        if not os.path.exists(video_path):
            logging.error(f"Video file does not exist: {video_path}")
            raise ValueError(f"Video file does not exist: {video_path}")
        
        duration = get_video_duration(video_path)
        if end_time > duration:
            logging.warning(f"Requested end_time ({end_time}s) exceeds video duration ({duration}s). Setting end_time to {duration}s.")
            end_time = duration
        
        # Initialize video capture for initial display
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            raise ValueError("Cannot open video file")
        
        # Display video until chessboard is detected or manually selected
        window_name = "Chessboard Detection (Press 'q' to select corners manually)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        board_detected = False
        board_points = None
        
        while cap.isOpened() and not board_detected:
            ret, frame = cap.read()
            if not ret:
                break
            
            crop, points = extract_digital_board(frame, debug=True)
            if points is not None:
                display_frame = frame.copy()
                points_int = points.astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(display_frame, [points_int], True, (0, 255, 0), 2)
                cv2.putText(display_frame, "Chessboard detected. Press 'a' to accept, 'c' for hardcoded ROI, 'm' for manual ROI", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window_name, display_frame)
                
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('a'):
                        board_detected = True
                        board_points = points
                        logging.info("Accepted automatically detected chessboard points")
                        break
                    elif key == ord('c'):
                        board_detected = True
                        board_points = np.array([
                            [1280, 240],  # top left
                            [1879, 240],  # top right
                            [1879, 836],  # bottom right
                            [1280, 836]   # bottom left
                        ], dtype="float32")
                        logging.info("Using hardcoded chessboard points")
                        break
                    elif key == ord('m'):
                        board_points = select_board_corners(frame)
                        board_detected = True
                        logging.info("Manually selected chessboard points")
                        break
            else:
                logging.info("Automatic chessboard detection failed")
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    board_points = select_board_corners(frame)
                    board_detected = True
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not board_detected:
            logging.info("No chessboard detected automatically, using hardcoded points")
            board_points = np.array([
                [1280, 240],  # top left
                [1879, 240],  # top right
                [1879, 836],  # bottom right
                [1280, 836]   # bottom left
            ], dtype="float32")
        
        # Extract frames for processing
        frames, frame_times, fps = extract_frames_in_duration(video_path, start_time, end_time, frame_interval)
        logging.info(f"Extracted {len(frames)} frames from {start_time}s to {end_time}s")
        
        # Initialize video writer
        output_path = "annotated_chess_moves.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_interval, (width, height))
        
        templates = load_templates()
        
        fen_results = []
        prev_board = None
        prev_fen = None
        board_history = []  # Store previous board states
        window_size = max(3, int(0.5 / frame_interval))  # Adjust window size based on frame interval
        
        for i, (frame, frame_time) in enumerate(zip(frames, frame_times)):
            logging.info(f"Processing frame at {frame_time:.2f} seconds (frame {i})")
            
            global match_result
            match_result = []
            
            try:
                crop = frame
                board, M = warp_board(crop, board_points)
                frame_debug_dir = f'./debug_frames/frame_{i:03d}'
                os.makedirs(frame_debug_dir, exist_ok=True)
                cv2.imwrite(f'{frame_debug_dir}/warped_board.png', board)

                squares, square_names, square_positions = split_into_squares(board, debug_dir=frame_debug_dir)
                fen, curr_board = generate_fen(squares, square_names, templates, frame_idx=i, debug=True)
                
                if fen is None:
                    logging.warning(f"Skipping frame {i} at {frame_time:.2f}s due to invalid FEN")
                    out.write(frame)
                    continue
                
                logging.info(f"Generated FEN at {frame_time:.2f}s: {fen}")
                
                moves = []
                if prev_fen is not None and fen != prev_fen:
                    moves = detect_movement(prev_board, curr_board, board_history, window_size)
                    logging.info(f"Detected moves at {frame_time:.2f}s: {moves}")
                
                annotated_frame = annotate_frame(frame, moves, frame_time, board_points, M)
                
                cv2.imshow(window_name, annotated_frame)
                if moves:
                    cv2.putText(annotated_frame, "Move detected! Press any key to continue.", 
                                (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        logging.info("User terminated video display with 'q' key")
                        break
                else:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        logging.info("User terminated video display with 'q' key")
                        break
                
                out.write(annotated_frame)
                
                fen_results.append((frame_time, fen, moves))
                prev_board = curr_board
                prev_fen = fen
                board_history.append(curr_board)
                if len(board_history) > window_size:
                    board_history.pop(0)
            
            except ValueError as e:
                logging.error(f"Processing error for frame at {frame_time:.2f}s: {e}")
                out.write(frame)
                continue
        
        print("\nFEN and Move Results:")
        for frame_time, fen, moves in fen_results:
            print(f"Time {frame_time:.2f}s: {fen}")
            if moves:
                print(f"  Moves: {', '.join(moves)}")

    except Exception as e:
        logging.error(f"Video processing error: {e}")
        print(f"Failed to process video: {e}")
    finally:
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(start_time=250, end_time=300, frame_interval=0.25)
