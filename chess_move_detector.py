import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import argparse

# Chessboard constants
BOARD_SIZE = 8
SQUARE_SIZE = 50  # Approximate size for transformed board
CHESSBOARD_CORNERS = (7, 7)  # Inner corners for 8x8 board

def find_chessboard_corners(image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find chessboard corners in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_CORNERS, None)
    if ret:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        return corners, gray
    return None, None

def get_perspective_transform(corners: np.ndarray, size: int = SQUARE_SIZE * BOARD_SIZE) -> Optional[np.ndarray]:
    """Calculate perspective transform to warp chessboard to a square grid."""
    corners = corners.reshape(-1, 2)
    top_left = corners[0]
    top_right = corners[CHESSBOARD_CORNERS[0] - 1]
    bottom_left = corners[-CHESSBOARD_CORNERS[0]]
    bottom_right = corners[-1]
    
    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst_pts = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return matrix

def divide_board(warped: np.ndarray) -> List[np.ndarray]:
    """Divide warped chessboard into 8x8 squares."""
    squares = []
    square_size = warped.shape[0] // BOARD_SIZE
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            square = warped[i * square_size:(i + 1) * square_size,
                           j * square_size:(j + 1) * square_size]
            squares.append(square)
    return squares

def detect_piece(square: np.ndarray, color_image: np.ndarray, square_idx: int, matrix: np.ndarray, frame: np.ndarray) -> Optional[Dict]:
    """Detect and classify a piece in a square."""
    # Convert square index to board coordinates
    row = square_idx // BOARD_SIZE
    col = square_idx % BOARD_SIZE
    square_size = SQUARE_SIZE
    
    # Map square back to original image for color analysis
    warped_pts = np.float32([
        [col * square_size, row * square_size],
        [(col + 1) * square_size, row * square_size],
        [(col + 1) * square_size, (row + 1) * square_size],
        [col * square_size, (row + 1) * square_size]
    ])
    original_pts = cv2.perspectiveTransform(np.array([warped_pts]), np.linalg.inv(matrix))[0]
    
    # Extract region from original color image
    pts = original_pts.astype(np.int32)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    color_square = frame[y_min:y_max, x_min:x_max]
    
    if color_square.size == 0:
        return None
    
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(color_square, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for white and black pieces (adjust based on pieces)
    white_lower = np.array([0, 0, 150])  # High value for white
    white_upper = np.array([180, 50, 255])
    black_lower = np.array([0, 0, 0])    # Low value for black
    black_upper = np.array([180, 255, 100])
    
    # Create masks for white and black pieces
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    black_mask = cv2.inRange(hsv, black_lower, black_upper)
    
    # Find contours in masks
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check for significant contours
    min_area = 100  # Minimum contour area to consider as a piece
    piece_info = None
    
    if white_contours:
        max_contour = max(white_contours, key=cv2.contourArea, default=None)
        if cv2.contourArea(max_contour) > min_area:
            piece_info = {"color": "white", "type": "unknown"}
    
    elif black_contours:
        max_contour = max(black_contours, key=cv2.contourArea, default=None)
        if cv2.contourArea(max_contour) > min_area:
            piece_info = {"color": "black", "type": "unknown"}
    
    return piece_info

def detect_change(prev_squares: List[np.ndarray], curr_squares: List[np.ndarray]) -> Optional[Tuple[int, int]]:
    """Detect changes between two sets of squares."""
    changes = []
    for i in range(len(prev_squares)):
        diff = cv2.absdiff(prev_squares[i], curr_squares[i])
        mean_diff = np.mean(diff)
        if mean_diff > 20:  # Threshold for significant change
            changes.append(i)
    
    if len(changes) == 2:  # Expect two squares to change (from and to)
        return changes[0], changes[1]
    return None

def index_to_notation(index: int) -> str:
    """Convert square index to chess notation (e.g., 0 -> a8, 63 -> h1)."""
    row = 7 - (index // BOARD_SIZE)
    col = index % BOARD_SIZE
    files = 'abcdefgh'
    return f"{files[col]}{row + 1}"

def process_video(source: str):
    """Process video feed or clip to detect chess moves and pieces."""
    cap = cv2.VideoCapture(source if source else 0)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source or 'webcam'}")
        return

    prev_squares = None
    prev_frame = None
    matrix = None
    board_state = [None] * (BOARD_SIZE * BOARD_SIZE)  # Track piece in each square

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Find chessboard corners
        corners, gray = find_chessboard_corners(frame)
        if corners is not None:
            # Draw corners for visualization
            cv2.drawChessboardCorners(frame, CHESSBOARD_CORNERS, corners, True)
            
            # Get perspective transform
            matrix = get_perspective_transform(corners)
            warped = cv2.warpPerspective(gray, matrix, (SQUARE_SIZE * BOARD_SIZE, SQUARE_SIZE * BOARD_SIZE))
            
            # Divide into squares
            curr_squares = divide_board(warped)

            # Detect pieces in each square
            for i in range(len(curr_squares)):
                piece_info = detect_piece(curr_squares[i], frame, i, matrix, frame)
                board_state[i] = piece_info

            # Detect changes if previous frame exists
            if prev_squares is not None:
                change = detect_change(prev_squares, curr_squares)
                if change:
                    from_idx, to_idx = change
                    from_notation = index_to_notation(from_idx)
                    to_notation = index_to_notation(to_idx)
                    piece = board_state[from_idx] or {"color": "unknown", "type": "unknown"}
                    print(f"Move detected: {piece['color']} {piece['type']} from {from_notation} to {to_notation}")

            prev_squares = curr_squares
            prev_frame = gray.copy()

            # Display board state (for debugging)
            for i, piece in enumerate(board_state):
                if piece:
                    pos = index_to_notation(i)
                    print(f"Square {pos}: {piece['color']} {piece['type']}")

        # Display the frame
        cv2.imshow('Chessboard', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Detect chess moves and pieces from video feed or clip.")
    parser.add_argument("--video", type=str, default="", help="Path to video clip (leave empty for webcam).")
    args = parser.parse_args()

    # Process video source (webcam or video file)
    process_video(args.video)

if __name__ == "__main__":
    main()
