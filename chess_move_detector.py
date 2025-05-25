import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
import argparse
import os
import tempfile
from pytube import YouTube

# Chessboard constants
BOARD_SIZE = 8
SQUARE_SIZE = 50  # Approximate size for transformed board
CHESSBOARD_CORNERS = (7, 7)  # Inner corners for 8x8 board

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image to improve chessboard detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def find_chessboard_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect the region of interest (ROI) containing the digital chessboard, prioritizing the right side."""
    gray = preprocess_image(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    image_width = image.shape[1]
    candidates = []
    
    # Filter for rectangular contours (potential chessboard)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:  # Check top 10 largest contours
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:  # Quadrilateral
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = w * h
            # Prioritize square-like regions on the right side, typical for digital boards
            if (0.8 < aspect_ratio < 1.2 and 
                area > 0.05 * image.shape[0] * image.shape[1] and  # Not too small
                area < 0.5 * image.shape[0] * image.shape[1] and   # Not too large (avoid physical board)
                x > image_width / 2):  # Right side of the frame
                candidates.append((x, y, w, h))
    
    # Sort candidates by x-coordinate (rightmost first)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0] if candidates else None

def find_chessboard_corners(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find chessboard corners in the image or ROI."""
    if roi:
        x, y, w, h = roi
        cropped = image[y:y+h, x:x+w]
        if cropped.size == 0:
            return None, None
    else:
        cropped = image
    
    gray = preprocess_image(cropped)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_CORNERS, flags=flags)
    if ret:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        if roi:
            corners += np.array([x, y], dtype=np.float32)
        return corners, gray
    return None, None

def get_perspective_transform(corners: np.ndarray, size: int = SQUARE_SIZE * BOARD_SIZE) -> Optional[np.ndarray]:
    """Calculate perspective transform to warp chessboard."""
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
    row = square_idx // BOARD_SIZE
    col = square_idx % BOARD_SIZE
    square_size = SQUARE_SIZE
    
    warped_pts = np.float32([
        [col * square_size, row * square_size],
        [(col + 1) * square_size, row * square_size],
        [(col + 1) * square_size, (row + 1) * square_size],
        [col * square_size, (row + 1) * square_size]
    ])
    original_pts = cv2.perspectiveTransform(np.array([warped_pts]), np.linalg.inv(matrix))[0]
    
    pts = original_pts.astype(np.int32)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    color_square = frame[y_min:y_max, x_min:x_max]
    
    if color_square.size == 0:
        return None
    
    hsv = cv2.cvtColor(color_square, cv2.COLOR_BGR2HSV)
    white_lower = np.array([0, 0, 150])
    white_upper = np.array([180, 50, 255])
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 100])
    
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    black_mask = cv2.inRange(hsv, black_lower, black_upper)
    
    white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = 100
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
        if mean_diff > 20:
            changes.append(i)
    
    if len(changes) == 2:
        return changes[0], changes[1]
    return None

def index_to_notation(index: int) -> str:
    """Convert square index to chess notation."""
    row = 7 - (index // BOARD_SIZE)
    col = index % BOARD_SIZE
    files = 'abcdefgh'
    return f"{files[col]}{row + 1}"

def download_youtube_video(url: str) -> str:
    """Download a YouTube video and return the temporary file path."""
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            raise ValueError("No suitable MP4 stream found.")
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"youtube_chess_{yt.video_id}.mp4")
        stream.download(output_path=temp_dir, filename=f"youtube_chess_{yt.video_id}.mp4")
        print(f"Downloaded YouTube video to: {temp_file}")
        return temp_file
    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        return ""

def process_video(source: str, is_youtube: bool = False, debug_dir: str = "debug_frames", manual_roi: Optional[List[int]] = None):
    """Process video feed, local file, or YouTube video to detect chess moves and pieces."""
    temp_file = None
    if is_youtube:
        temp_file = download_youtube_video(source)
        if not temp_file or not os.path.exists(temp_file):
            print("Failed to download YouTube video.")
            return
        source = temp_file

    cap = cv2.VideoCapture(source if source else 0)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source or 'webcam'}")
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        return

    os.makedirs(debug_dir, exist_ok=True)
    frame_count = 0

    prev_squares = None
    prev_frame = None
    matrix = None
    board_state = [None] * (BOARD_SIZE * BOARD_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Detect ROI
        roi = manual_roi
        if not roi:
            roi = find_chessboard_region(frame)
        
        if roi:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw ROI
            roi_frame = frame[y:y+h, x:x+w]
            debug_roi_path = os.path.join(debug_dir, f"roi_frame_{frame_count}.jpg")
            cv2.imwrite(debug_roi_path, roi_frame)
            print(f"Saved ROI debug frame: {debug_roi_path}")
        else:
            roi_frame = frame
            print(f"Frame {frame_count}: Could not detect chessboard region.")

        corners, gray = find_chessboard_corners(frame, roi)
        if corners is not None:
            cv2.drawChessboardCorners(frame, CHESSBOARD_CORNERS, corners, True)
            matrix = get_perspective_transform(corners)
            warped = cv2.warpPerspective(gray, matrix, (SQUARE_SIZE * BOARD_SIZE, SQUARE_SIZE * BOARD_SIZE))
            curr_squares = divide_board(warped)

            for i in range(len(curr_squares)):
                piece_info = detect_piece(curr_squares[i], frame, i, matrix, frame)
                board_state[i] = piece_info

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

            for i, piece in enumerate(board_state):
                if piece:
                    pos = index_to_notation(i)
                    print(f"Square {pos}: {piece['color']} {piece['type']}")
        else:
            print(f"Frame {frame_count}: Chessboard not detected.")
            debug_path = os.path.join(debug_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(debug_path, frame)
            print(f"Saved debug frame: {debug_path}")

        cv2.imshow('Chessboard', frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)

def main():
    parser = argparse.ArgumentParser(description
