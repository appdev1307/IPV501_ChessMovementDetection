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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clipLimit for better contrast
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduced blur to preserve edges
    return gray

def check_grid_pattern(image: np.ndarray, roi: Tuple[int, int, int, int]) -> bool:
    """Check if the ROI contains an 8x8 grid pattern typical of a chessboard."""
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w]
    if cropped.size == 0:
        return False
    
    # Convert to binary image
    _, binary = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find horizontal and vertical lines
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=w//4, maxLineGap=10)
    
    if lines is None:
        return False
    
    horizontal_lines = 0
    vertical_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 10 or angle > 170:  # Horizontal line
            horizontal_lines += 1
        elif 80 < angle < 100:  # Vertical line
            vertical_lines += 1
    
    # Expect around 7-9 lines for an 8x8 grid (inner lines)
    return 6 <= horizontal_lines <= 10 and 6 <= vertical_lines <= 10

def find_chessboard_region(image: np.ndarray, debug_dir: str = "debug_frames") -> Optional[Tuple[int, int, int, int]]:
    """Detect the region of interest (ROI) containing the digital chessboard, prioritizing the right side."""
    os.makedirs(debug_dir, exist_ok=True)
    
    # Preprocess with stronger contrast and edge-preserving blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))  # Even stronger contrast
    gray = clahe.apply(gray)
    # Enhance contrast on the right 50%
    right_start = int(image.shape[1] * 0.5)
    right_half = gray[:, right_start:]
    gray[:, right_start:] = cv2.equalizeHist(right_half)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)  # Preserve edges, reduce noise
    
    # Save preprocessed image
    preprocessed_path = os.path.join(debug_dir, "preprocessed.jpg")
    cv2.imwrite(preprocessed_path, gray)
    print(f"Saved preprocessed image: {preprocessed_path}")
    
    # Try Otsu's thresholding for better grid detection
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Save thresholded images
    thresh_path = os.path.join(debug_dir, "thresholded.jpg")
    right_thresh_path = os.path.join(debug_dir, "thresholded_right.jpg")
    cv2.imwrite(thresh_path, thresh)
    cv2.imwrite(right_thresh_path, thresh[:, right_start:])
    print(f"Saved thresholded image: {thresh_path}")
    print(f"Saved right-half thresholded image: {right_thresh_path}")
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None
    
    image_width = image.shape[1]
    image_height = image.shape[0]
    candidates = []
    
    # Draw all contours for debugging
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)
    contour_path = os.path.join(debug_dir, "contours.jpg")
    cv2.imwrite(contour_path, contour_image)
    print(f"Saved contour image: {contour_path}")
    
    print("Contour analysis:")
    for i, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:20]):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = w * h
            area_ratio = area / (image_width * image_height)
            if w * h > 10000:  # Lowered minimum area
                print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}, area_ratio={area_ratio:.3f}, aspect_ratio={aspect_ratio:.2f}")
                # Relaxed geometric filters
                if (0.7 < aspect_ratio < 1.3 and 
                    0.03 < area_ratio < 0.20 and 
                    x > image_width * 0.5 and  # Relaxed right-side focus
                    y < image_height * 0.7):
                    # Check for chessboard pattern
                    cropped = thresh[y:y+h, x:x+w]
                    # Save cropped region for debugging
                    cropped_path = os.path.join(debug_dir, f"cropped_contour_{i}.jpg")
                    cv2.imwrite(cropped_path, cropped)
                    print(f"Saved cropped contour {i}: {cropped_path}")
                    
                    pattern_check = check_chessboard_pattern(cropped)
                    print(f"Contour {i} pattern check: {pattern_check}")
                    if pattern_check:
                        candidates.append((x, y, w, h))
                        print(f"Contour {i} accepted as chessboard candidate.")
                    else:
                        print(f"Contour {i} rejected: No chessboard pattern.")
                else:
                    print(f"Contour {i} rejected: Failed geometric filters.")
    
    if not candidates:
        print("No suitable candidates found.")
        return None
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_roi = candidates[0]
    print(f"Selected ROI: {selected_roi}")
    return selected_roi

def find_chessboard_region(image: np.ndarray, debug_dir: str = "debug_frames") -> Optional[Tuple[int, int, int, int]]:
    """Detect the region of interest (ROI) containing the digital chessboard."""
    os.makedirs(debug_dir, exist_ok=True)
    
    # Image dimensions
    image_height, image_width = image.shape[:2]
    print(f"Image dimensions: {image_width}x{image_height}")
    
    # Dynamically calculate expected chessboard size (assuming it’s ~30% of the image width)
    expected_size = int(image_width * 0.3)  # ~548 for 1828 width
    expected_area_ratio = (expected_size * expected_size) / (image_width * image_height)  # ~0.09 for 1828x1044
    print(f"Expected chessboard size: {expected_size}x{expected_size}, expected area ratio: {expected_area_ratio:.3f}")
    
    # Preprocess with edge detection to highlight grid
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))  # Stronger contrast
    gray = clahe.apply(gray)
    edges = cv2.Canny(gray, 50, 150)  # Detect edges for grid lines
    gray = cv2.addWeighted(gray, 0.7, cv2.dilate(edges, None), 0.3, 0.0)  # Combine edges with gray
    
    # Save preprocessed image
    preprocessed_path = os.path.join(debug_dir, "preprocessed.jpg")
    cv2.imwrite(preprocessed_path, gray)
    print(f"Saved preprocessed image: {preprocessed_path}")
    
    # Hybrid thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Save thresholded images
    thresh_path = os.path.join(debug_dir, "thresholded.jpg")
    cv2.imwrite(thresh_path, thresh)
    print(f"Saved thresholded image: {thresh_path}")
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None
    
    # Draw all contours for debugging
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)
    contour_path = os.path.join(debug_dir, "contours.jpg")
    cv2.imwrite(contour_path, contour_image)
    print(f"Saved contour image: {contour_path}")
    
    print("Contour analysis:")
    candidates = []
    for i, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)[:20]):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = w * h
            area_ratio = area / (image_width * image_height)
            min_area = (expected_size * 0.25) ** 2  # ~10% of expected area
            if area > min_area:
                print(f"Contour {i}: x={x}, y={y}, w={w}, h={h}, area_ratio={area_ratio:.3f}, aspect_ratio={aspect_ratio:.2f}")
                # Relaxed geometric filters
                if (0.6 < aspect_ratio < 1.4 and  # Very relaxed aspect ratio
                    expected_area_ratio * 0.3 < area_ratio < expected_area_ratio * 3.0):  # Wide area range
                    # Check for chessboard pattern
                    cropped = thresh[y:y+h, x:x+w]
                    # Save cropped region for debugging
                    cropped_path = os.path.join(debug_dir, f"cropped_contour_{i}.jpg")
                    cv2.imwrite(cropped_path, cropped)
                    print(f"Saved cropped contour {i}: {cropped_path}")
                    
                    pattern_check = check_chessboard_pattern(cropped)
                    print(f"Contour {i} pattern check: {pattern_check}")
                    if pattern_check:
                        candidates.append((x, y, w, h))
                        print(f"Contour {i} accepted as chessboard candidate.")
                    else:
                        print(f"Contour {i} rejected: No chessboard pattern.")
                else:
                    print(f"Contour {i} rejected: Failed geometric filters.")
    
    if not candidates:
        print("No suitable candidates found.")
        return None
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected_roi = candidates[0]
    print(f"Selected ROI: {selected_roi}")
    return selected_roi

def check_chessboard_pattern(image: np.ndarray) -> bool:
    """Check for alternating light/dark squares indicative of a chessboard."""
    if image.size == 0:
        return False
    
    # Resize to a manageable size
    h, w = image.shape
    scale = 100 / max(h, w)
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Count alternating regions
    rows, cols = resized.shape
    dark_count = 0
    light_count = 0
    for i in range(0, rows, max(1, rows//8)):
        for j in range(0, cols, max(1, cols//8)):
            region = resized[i:i+max(1, rows//8), j:j+max(1, cols//8)]
            mean_val = np.mean(region)
            if mean_val < 128:
                dark_count += 1
            else:
                light_count += 1
    
    # Debug pattern check
    total_squares = dark_count + light_count
    print(f"Pattern check - Dark squares: {dark_count}, Light squares: {light_count}, Total: {total_squares}")
    
    if total_squares < 8:  # Very relaxed minimum
        return False
    return abs(dark_count - light_count) < total_squares * 0.5  # Very relaxed balance

# Rest of the script remains unchanged (find_chessboard_corners, get_perspective_transform, etc.)

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

def process_image(image_path: str, debug_dir: str = "debug_frames", manual_roi: Optional[List[int]] = None):
    """Process a single image to detect chessboard and pieces."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    os.makedirs(debug_dir, exist_ok=True)

    roi = manual_roi
    if not roi:
        roi = find_chessboard_region(frame, debug_dir)
    
    if roi:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_frame = frame[y:y+h, x:x+w]
        debug_roi_path = os.path.join(debug_dir, "roi_image.jpg")
        cv2.imwrite(debug_roi_path, roi_frame)
        print(f"Saved ROI debug image: {debug_roi_path}")
    else:
        roi_frame = frame
        print("Could not detect chessboard region.")

    corners, gray = find_chessboard_corners(frame, roi)
    if corners is not None:
        cv2.drawChessboardCorners(frame, CHESSBOARD_CORNERS, corners, True)
        matrix = get_perspective_transform(corners)
        warped = cv2.warpPerspective(gray, matrix, (SQUARE_SIZE * BOARD_SIZE, SQUARE_SIZE * BOARD_SIZE))
        curr_squares = divide_board(warped)

        board_state = [None] * (BOARD_SIZE * BOARD_SIZE)
        for i in range(len(curr_squares)):
            piece_info = detect_piece(curr_squares[i], frame, i, matrix, frame)
            board_state[i] = piece_info

        for i, piece in enumerate(board_state):
            if piece:
                pos = index_to_notation(i)
                print(f"Square {pos}: {piece['color']} {piece['type']}")
    else:
        print("Chessboard not detected.")
        debug_path = os.path.join(debug_dir, "image.jpg")
        cv2.imwrite(debug_path, frame)
        print(f"Saved debug image: {debug_path}")

    cv2.imshow('Chessboard', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

        roi = manual_roi
        if not roi:
            roi = find_chessboard_region(frame, debug_dir)
        
        if roi:
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
    parser = argparse.ArgumentParser(description="Detect chess moves and pieces from image, video feed, local file, or YouTube.")
    parser.add_argument("--image", type=str, default="", help="Path to image file.")
    parser.add_argument("--video", type=str, default="", help="Path to local video file (leave empty for webcam).")
    parser.add_argument("--youtube", type=str, default="", help="YouTube video URL.")
    parser.add_argument("--debug-dir", type=str, default="debug_frames", help="Directory to save debug frames.")
    parser.add_argument("--roi", type=int, nargs=4, help="Manual ROI coordinates [x, y, width, height].")
    args = parser.parse_args()

    if args.image:
        process_image(args.image, debug_dir=args.debug_dir, manual_roi=args.roi)
    elif args.youtube:
        process_video(args.youtube, is_youtube=True, debug_dir=args.debug_dir, manual_roi=args.roi)
    else:
        process_video(args.video, is_youtube=False, debug_dir=args.debug_dir, manual_roi=args.roi)

if __name__ == "__main__":
    main()
