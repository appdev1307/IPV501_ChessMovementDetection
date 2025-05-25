import cv2
import numpy as np
from IPython.display import display, Image
import matplotlib.pyplot as plt

def find_chessboard_corners(frame):
    """Find the corners of the chessboard in the frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return ret, corners

def get_perspective_transform(corners, frame_shape):
    """Calculate the perspective transform matrix for the chessboard."""
    # Define the destination points for the transform
    board_size = 400  # Size of the output board
    dst_points = np.float32([[0, 0], [board_size, 0], [0, board_size], [board_size, board_size]])
    
    # Get the four corners of the chessboard
    src_points = np.float32([corners[0][0], corners[6][0], corners[42][0], corners[48][0]])
    
    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

def get_square_coordinates(board_size=400):
    """Generate coordinates for all 64 squares on the chessboard."""
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

def detect_changes(prev_board, curr_board, threshold=30):
    """Detect changes between two consecutive frames."""
    diff = cv2.absdiff(prev_board, curr_board)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    return thresh

def process_chess_video(video_source=0):
    """Main function to process the video feed and detect chess movements."""
    cap = cv2.VideoCapture(video_source)
    prev_board = None
    squares = get_square_coordinates()
    
    board_states = []  # List to store board images after each move
    move_list = []     # List to store move notations
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Find chessboard corners
        ret, corners = find_chessboard_corners(frame)
        if not ret:
            continue
            
        # Get perspective transform
        matrix = get_perspective_transform(corners, frame.shape)
        board = cv2.warpPerspective(frame, matrix, (400, 400))
        
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
                move = f"{changed_squares[0]} to {changed_squares[1]}"
                print(f"Move detected: {move}")
                move_list.append(move)
                board_states.append(board.copy())  # Save the board image
        
        prev_board = board.copy()
        
        # Display the processed board
        cv2.imshow('Chess Board', board)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    # cv2.destroyAllWindows()
    
    # After processing, board_states and move_list contain the history for visualization
    return board_states, move_list

if __name__ == "__main__":
    process_chess_video() 