import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Match result
match_result = []

# Erosion kernel
EROSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

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
    """Extract and display frames from a video within a specified duration at given intervals."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file: {video_path}")
        raise ValueError("Cannot open video file")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logging.error("Invalid FPS value in video")
        raise ValueError("Invalid FPS value")
    
    # Validate duration
    duration = get_video_duration(video_path)
    if start_time < 0 or end_time > duration or start_time >= end_time:
        logging.error(f"Invalid duration: start_time={start_time}s, end_time={end_time}s, video_duration={duration}s")
        raise ValueError(f"Invalid duration: start_time={start_time}s, end_time={end_time}s, video_duration={duration}s")
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_step = max(1, int(frame_interval * fps))  # Ensure at least 1 frame step
    
    frames = []
    frame_times = []
    
    # Create a window for displaying frames
    window_name = "Chessboard Video Frame"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    for frame_num in range(start_frame, end_frame + 1, frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Failed to extract frame {frame_num} from video")
            continue
        
        # Display the frame
        cv2.imshow(window_name, frame)
        frame_time = frame_num / fps
        logging.info(f"Displaying frame at {frame_time:.2f} seconds (frame {frame_num})")
        
        # Add a delay and check for 'q' key to exit
        if cv2.waitKey(100) & 0xFF == ord('q'):  # 100ms delay, exit on 'q'
            logging.info("User terminated video display with 'q' key")
            break
        
        frames.append(frame)
        frame_times.append(frame_time)
    
    cap.release()
    cv2.destroyWindow(window_name)  # Clean up the display window
    
    if not frames:
        logging.error(f"No frames extracted in the specified duration: {start_time}s to {end_time}s")
        raise ValueError("No frames extracted in the specified duration")
    
    return frames, frame_times

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def extract_digital_board(image, debug=False):
    print('Crop a broader region and detect a rectangular board-like contour')
    h, w = image.shape[:2]
    crop = image[0:h, 0:w]
    if crop.size == 0:
        raise ValueError("Cropped image is empty")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_img = crop.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
        plt.ion()
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Debug: Detected Contours")
        plt.axis("off")
        os.makedirs('./debug_frames', exist_ok=True)
        plt.savefig('./debug_frames/_dplot.png')
        plt.close()

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) >= 4 and cv2.contourArea(c) > 5000:
            points = approx.reshape(-1, 2)
            print(f"Detected points: {points}")
            return crop, points

    print("No valid board contour found")
    return crop, None

def load_templates(template_dir="templates", debug_dir="debug_output", debug=False):
    pieces = ["P", "N", "B", "R", "Q", "K", "pb", "nb", "bb", "rb", "qb", "kb"]
    templates = {}
    os.makedirs(debug_dir, exist_ok=True)

    for p in pieces:
        path = os.path.join(template_dir, f"{p}.png")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.error(f"Template for {p} not found at {path}")
            raise ValueError(f"Template for {p} not found at {path}")

        resized = cv2.resize(img, (80, 80))
        eroded = cv2.erode(resized, EROSION_KERNEL, iterations=1)
        templates[p] = eroded

        if debug:
            cv2.imwrite(os.path.join(debug_dir, f"{p}_eroded.png"), eroded)

    return templates

def match_piece(square_img, img_name, templates, threshold=0.6, debug=False):
    if square_img.size == 0 or square_img.shape[0] == 0 or square_img.shape[1] == 0:
        logging.warning(f"Empty square image: {img_name}")
        return None

    square_gray = square_img if len(square_img.shape) == 2 else cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    square_resized = cv2.resize(square_gray, (80, 80))
    square_resized = cv2.erode(square_resized, EROSION_KERNEL, iterations=1)

    max_val = 0
    best_match = None

    for piece, template in templates.items():
        try:
            res = cv2.matchTemplate(square_resized, template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > max_val:
                max_val = val
                best_match = piece
        except cv2.error as e:
            logging.error(f"matchTemplate error for {piece} in {img_name}: {e}")
            continue

    if max_val >= threshold:
        if debug:
            os.makedirs('./debug_frames/match', exist_ok=True)
            vis = np.hstack([square_resized, templates[best_match]])
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis_bgr, f"{best_match} ({max_val:.2f})", (5, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(f'./debug_frames/match/{img_name}_match.png', vis_bgr)
            match_result.append(f"{img_name} match ={best_match}")
        return best_match
    return None

def warp_board(crop, points):
    rect = order_points(points)
    dst = np.array([
        [0, 0],
        [551, 0],
        [551, 551],
        [0, 551]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(crop, M, (552, 552))

    if warped.size == 0 or warped.shape[0] == 0 or warped.shape[1] == 0:
        raise ValueError("Warped image is empty or invalid")

    return warped

def split_into_squares(board_img, debug_dir="./debug_frames"):
    os.makedirs(debug_dir, exist_ok=True)
    squares = []
    square_names = []
    height, width = board_img.shape[:2]
    dy, dx = 69, 69

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
            if square.size == 0 or square.shape[0] < dy or square.shape[1] < dx:
                logging.warning(f"Empty or undersized square at row {row}, col {col}")
                continue
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if len(square.shape) == 3 else square
            resized = cv2.resize(gray, (68, 68))
            name = f'square_r{row}_c{col}.png'
            cv2.imwrite(os.path.join(debug_dir, name), resized)
            squares.append(resized)
            square_names.append(name)
    return squares, square_names

def generate_fen(squares, square_names, templates, debug=False):
    board = [['' for _ in range(8)] for _ in range(8)]
    fen_map = {'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
               'pb': 'p', 'nb': 'n', 'bb': 'b', 'rb': 'r', 'qb': 'q', 'kb': 'k'}

    for i, (square, name) in enumerate(zip(squares, square_names)):
        row, col = i // 8, i % 8
        piece = match_piece(square, name, templates, debug=debug)
        board[row][col] = fen_map.get(piece, '') if piece else ''

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
    return fen

def main(video_path, start_time=0, end_time=10, frame_interval=1.0, use_dynamic_board_detection=False):
    try:
        # Validate video file existence
        if not os.path.exists(video_path):
            logging.error(f"Video file does not exist: {video_path}")
            raise ValueError(f"Video file does not exist: {video_path}")
        
        # Validate duration
        duration = get_video_duration(video_path)
        if end_time > duration:
            logging.warning(f"Requested end_time ({end_time}s) exceeds video duration ({duration}s). Setting end_time to {duration}s.")
            end_time = duration
        
        # Extract and display frames within the specified duration
        frames, frame_times = extract_frames_in_duration(video_path, start_time, end_time, frame_interval)
        logging.info(f"Extracted {len(frames)} frames from {start_time}s to {end_time}s")
        
        # Load templates once
        templates = load_templates("templates", debug_dir="debug_output", debug=True)
        
        # Process each frame
        fen_results = []
        for i, (frame, frame_time) in enumerate(zip(frames, frame_times)):
            logging.info(f"Processing frame at {frame_time:.2f} seconds (frame {i})")
            
            # Reset match_result for each frame
            global match_result
            match_result = []
            
            try:
                # Try dynamic board detection if enabled
                if use_dynamic_board_detection:
                    crop, points = extract_digital_board(frame, debug=True)
                    if points is None:
                        logging.warning(f"No board detected in frame at {frame_time:.2f}s, skipping")
                        continue
                else:
                    # Use hardcoded points
                    points = np.array([
                        [1280, 240], # top left
                        [1879, 240], # top right
                        [1879, 836],  # bottem right
                        [1280, 836] # bottom left
                    ], dtype="float32")
                    crop = frame

                logging.info(f"Warping board with points: {points.tolist()}")
                board = warp_board(crop, points)
                logging.info(f"Warped board shape: {board.shape}")
                frame_debug_dir = f'./debug_frames/frame_{i:03d}'
                os.makedirs(frame_debug_dir, exist_ok=True)
                cv2.imwrite(f'{frame_debug_dir}/warped_board.png', board)

                squares, square_names = split_into_squares(board, debug_dir=frame_debug_dir)
                fen = generate_fen(squares, square_names, templates, debug=True)
                logging.info(f"Generated FEN at {frame_time:.2f}s: {fen}")
                """fen_results.append((frame_time, fen))

                # Unit test for each frame
                match_UT1 = [
                    "square_r0_c4.png match =qb",
                    "square_r0_c5.png match =rb",
                    "square_r0_c6.png match =kb",
                    "square_r1_c2.png match =R",
                    "square_r1_c4.png match =bb",
                    "square_r1_c5.png match =pb",
                    "square_r1_c6.png match =pb",
                    "square_r1_c7.png match =pb",
                    "square_r2_c0.png match =bb",
                    "square_r2_c1.png match =pb",
                    "square_r3_c0.png match =pb",
                    "square_r3_c3.png match =rb",
                    "square_r4_c0.png match =P",
                    "square_r4_c4.png match =P",
                    "square_r5_c1.png match =Q",
                    "square_r5_c5.png match =N",
                    "square_r5_c6.png match =P",
                    "square_r5_c7.png match =B",
                    "square_r6_c1.png match =P",
                    "square_r6_c5.png match =P",
                    "square_r6_c7.png match =P",
                    "square_r7_c6.png match =K"
                ]

                max_len = max(len(match_result), len(match_UT1))
                for j in range(max_len):
                    item1 = match_result[j] if j < len(match_result) else "<missing>"
                    item2 = match_UT1[j] if j < len(match_UT1) else "<missing>"
                    if item1 != item2:
                        print(f"Frame at {frame_time:.2f}s, Difference at index {j}: match_result = '{item1}', match_UT1 = '{item2}'")
            """
            except ValueError as e:
                logging.error(f"Processing error for frame at {frame_time:.2f}s: {e}")
                continue
        
        # Print all FEN results
        print("\nFEN Results:")
        for frame_time, fen in fen_results:
            print(f"Time {frame_time:.2f}s: {fen}")

    except Exception as e:
        logging.error(f"Video processing error: {e}")
        print(f"Failed to process video: {e}")
    finally:
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the local video file
    video_path = "video.mp4.mkv"  # Replace with the actual path to your downloaded clip
    start_time = 300  # Start at 0 seconds
    end_time = 305   # End at 10 seconds
    frame_interval = 1.0  # Process one frame per second
    use_dynamic_board_detection = False  # Set to True to use dynamic board detection
    main(video_path, start_time, end_time, frame_interval, use_dynamic_board_detection)
