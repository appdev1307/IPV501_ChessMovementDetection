import cv2
import numpy as np
from IPython.display import display, Image
import matplotlib.pyplot as plt
from chess_rules import ChessMove, ChessRules
import tkinter as tk
from tkinter import filedialog
import os
import logging
import datetime
import pytesseract
from PIL import Image

# Set up logging
def setup_logging():
    """Set up logging configuration."""
    log_filename = f"chess_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # Set the root logger to DEBUG level
    logging.getLogger().setLevel(logging.DEBUG)
    return logging.getLogger(__name__)

logger = setup_logging()

def select_video_file():
    """Open a file dialog to select a video file."""
    logger.info("Opening file dialog to select video")
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
    )
    if file_path:
        logger.info(f"Selected video file: {file_path}")
    else:
        logger.warning("No video file selected")
    return file_path

def find_chessboard_corners(frame):
    """Find the corners of the chessboard in the frame."""
    logger.debug("Attempting to find chessboard corners")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    if ret:
        logger.debug("Chessboard corners found")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        logger.debug("No chessboard corners found in frame")
    return ret, corners

def get_perspective_transform(corners, frame_shape):
    """Calculate the perspective transform matrix for the chessboard."""
    logger.debug("Calculating perspective transform")
    board_size = 400
    dst_points = np.float32([[0, 0], [board_size, 0], [0, board_size], [board_size, board_size]])
    src_points = np.float32([corners[0][0], corners[6][0], corners[42][0], corners[48][0]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

def get_square_coordinates(board_size=400):
    """Generate coordinates for all 64 squares on the chessboard."""
    logger.debug("Generating square coordinates")
    square_size = board_size // 8
    squares = []
    for row in range(8):
        for col in range(8):
            x1 = col * square_size
            y1 = row * square_size
            x2 = x1 + square_size
            y2 = y1 + square_size
            squares.append(((x1, y1), (x2, y2)))
    return squares

def get_square_name(row, col):
    """Convert row and column indices to chess notation."""
    files = 'abcdefgh'
    ranks = '87654321'
    return f"{files[col]}{ranks[row]}"

def detect_piece_type(square_img):
    """Detect the type of piece in a square using template matching."""
    logger.debug("Attempting to detect piece type")
    # This is a placeholder - in a real implementation, you would use
    # computer vision techniques to identify the piece type
    return 'unknown'

def detect_changes(prev_board, curr_board, threshold=30):
    """Detect changes between two consecutive frames."""
    logger.debug("Detecting changes between frames")
    diff = cv2.absdiff(prev_board, curr_board)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def process_chess_video(video_path=None):
    """Main function to process the video feed and detect chess movements."""
    logger.info("Starting chess video processing")
    
    if video_path is None:
        video_path = select_video_file()
        if not video_path:
            logger.error("No video file selected")
            return

    logger.info(f"Opening video file: {video_path}")
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video file")
            return
        
        # Log video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}, Total frames: {total_frames}")
        
    except Exception as e:
        logger.error(f"Error opening video file: {str(e)}", exc_info=True)
        return

    prev_board = None
    squares = get_square_coordinates()
    chess_rules = ChessRules()
    current_turn = 'white'  # White moves first
    
    board_states = []  # List to store board images after each move
    move_list = []     # List to store move notations
    
    frame_count = 0
    logger.info("Starting video processing loop")
    
    try:
        while True:
            frame_count += 1
            logger.debug(f"Processing frame {frame_count}")
            
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                logger.debug(f"Successfully read frame {frame_count}")
                
                # Find chessboard corners
                ret, corners = find_chessboard_corners(frame)
                if not ret:
                    logger.debug(f"Could not find chessboard in frame {frame_count}")
                    continue
                
                logger.debug(f"Found chessboard in frame {frame_count}")
                
                # Get perspective transform
                try:
                    matrix = get_perspective_transform(corners, frame.shape)
                    board = cv2.warpPerspective(frame, matrix, (400, 400))
                    logger.debug(f"Applied perspective transform to frame {frame_count}")
                except Exception as e:
                    logger.error(f"Error in perspective transform for frame {frame_count}: {str(e)}")
                    continue
                
                if prev_board is not None:
                    # Detect changes
                    changes = detect_changes(prev_board, board)
                    
                    # Analyze changes in each square
                    changed_squares = []
                    for i, ((x1, y1), (x2, y2)) in enumerate(squares):
                        square_changes = changes[y1:y2, x1:x2]
                        if np.mean(square_changes) > 10:  # Threshold for significant change
                            row = i // 8
                            col = i % 8
                            changed_squares.append(get_square_name(row, col))
                    
                    # If exactly two squares changed, it's likely a move
                    if len(changed_squares) == 2:
                        from_square = changed_squares[0]
                        to_square = changed_squares[1]
                        logger.info(f"Potential move detected: {from_square} to {to_square}")
                        
                        # Try to detect piece type
                        piece_type = detect_piece_type(board[y1:y2, x1:x2])
                        
                        # Create move object
                        move = ChessMove(
                            piece_type=piece_type,
                            color=current_turn,
                            from_square=from_square,
                            to_square=to_square
                        )
                        
                        # Validate and make the move
                        if chess_rules.make_move(move):
                            move_notation = f"{current_turn} - {piece_type} - {from_square} - {to_square}"
                            logger.info(f"Valid move detected: {move_notation}")
                            move_list.append(move_notation)
                            board_states.append(board.copy())
                            
                            # Switch turns
                            current_turn = 'black' if current_turn == 'white' else 'white'
                        else:
                            logger.warning(f"Invalid move: {move.warning}")
                
                prev_board = board.copy()
                
                # Display the processed board
                cv2.imshow('Chess Board', board)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested to quit")
                    break
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}", exc_info=True)
                continue
                
    except Exception as e:
        logger.error(f"Error in main processing loop: {str(e)}", exc_info=True)
    finally:
        logger.info("Releasing video capture")
        cap.release()
        cv2.destroyAllWindows()
        
        # Create slideshow of the game
        logger.info("Creating game slideshow")
        create_game_slideshow(board_states, move_list)
        
        logger.info(f"Processing complete. Total frames processed: {frame_count}")
        logger.info(f"Total moves detected: {len(move_list)}")
        
        return board_states, move_list

def create_game_slideshow(board_states, move_list):
    """Create a slideshow of the chess game."""
    logger.info("Starting slideshow creation")
    if not board_states or not move_list:
        logger.warning("No moves recorded for slideshow")
        return

    plt.figure(figsize=(10, 8))
    for i, (board, move) in enumerate(zip(board_states, move_list)):
        logger.debug(f"Creating slide {i+1} for move: {move}")
        plt.clf()
        plt.imshow(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
        plt.title(f"Move {i+1}: {move}")
        plt.axis('off')
        plt.pause(2)  # Show each position for 2 seconds
    
    plt.close()
    logger.info("Slideshow creation complete")

def extract_moves_from_video(video_path, crop_area, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    moves_set = set()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            # Crop the move list area
            x1, y1, x2, y2 = crop_area
            move_img = frame[y1:y2, x1:x2]
            # Convert to PIL Image for pytesseract
            pil_img = Image.fromarray(cv2.cvtColor(move_img, cv2.COLOR_BGR2RGB))
            # OCR
            text = pytesseract.image_to_string(pil_img, lang='eng')
            # Parse moves (very basic, you can improve this)
            for line in text.splitlines():
                if any(char.isdigit() for char in line) and '.' in line:
                    moves_set.add(line.strip())
        frame_idx += 1

    cap.release()
    # Sort and print moves
    moves = sorted(list(moves_set))
    for move in moves:
        print(move)
    # Optionally, save to file
    with open('extracted_moves.txt', 'w', encoding='utf-8') as f:
        for move in moves:
            f.write(move + '\n')

# Example: You need to adjust these coordinates for your video!
# (left, top, right, bottom) of the move list area
crop_area = (600, 100, 800, 600)  # <-- Adjust these!

if __name__ == "__main__":
    try:
        logger.info("Starting chess detection program")
        process_chess_video()
        logger.info("Program completed successfully")
    except Exception as e:
        logger.error(f"Program failed with error: {str(e)}", exc_info=True) 