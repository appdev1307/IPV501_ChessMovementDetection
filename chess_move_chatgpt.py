import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def order_points(pts):
    """Order four points to form a rectangle: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left: min sum
    rect[2] = pts[np.argmax(s)]  # Bottom-right: max sum
    rect[1] = pts[np.argmin(diff)]  # Top-right: min diff
    rect[3] = pts[np.argmax(diff)]  # Bottom-left: max diff
    return rect

def extract_digital_board(image, debug=False):
    print('Crop a broader region and detect a rectangular board-like contour')
    h, w = image.shape[:2]
    crop = image[0:h, 0:w]  # Use full image
    if crop.size == 0:
        raise ValueError("Cropped image is empty")
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    #sharpened = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    # Define sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])

    # Apply filter
    #sharpened = cv2.filter2D(gray, -1, kernel)

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
    """Load and preprocess chess piece templates."""
    pieces = ["P", "N", "B", "R", "Q", "K", "pb", "nb", "bb", "rb", "qb", "kb"]
    templates = {}
    os.makedirs(debug_dir, exist_ok=True)

    for p in pieces:
        path = os.path.join(template_dir, f"{p}.png")
        img = cv2.imread(path)
        if img is None:
            logging.error(f"Template for {p} not found at {path}")
            raise ValueError(f"Template for {p} not found at {path}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 80])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        white_bg = np.full_like(gray, 255)
        piece_on_white = np.where(mask_clean == 255, gray, white_bg)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bordered = piece_on_white.copy()
        cv2.drawContours(bordered, contours, -1, color=255, thickness=1)
        resized = cv2.resize(bordered, (68, 68))
        templates[p] = resized

        if debug:
            cv2.imwrite(os.path.join(debug_dir, f"{p}_original.png"), img)
            cv2.imwrite(os.path.join(debug_dir, f"{p}_mask.png"), mask_clean)
            cv2.imwrite(os.path.join(debug_dir, f"{p}_resized.png"), resized)
            logging.info(f"Saved debug images for template {p}")

    return templates

def match_piece(square_img, img_name, templates, threshold=0.5, debug=False):
    """Match a chess piece in a square to a template with lower threshold for debugging."""
    if square_img.size == 0 or square_img.shape[0] == 0 or square_img.shape[1] == 0:
        logging.warning(f"Empty square image: {img_name}")
        return None

    square_gray = square_img if len(square_img.shape) == 2 else cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(square_gray, 200, 255, cv2.THRESH_BINARY_INV)
    white_bg = np.full_like(square_gray, 255)
    square_clean = np.where(mask == 255, square_gray, white_bg)
    square_resized = cv2.resize(square_clean, (68, 68))

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

    #logging.info(f"{img_name} match result: best={best_match}, score={max_val:.4f}")
    if max_val >= threshold:
        if debug:
            os.makedirs('./debug_frames/match', exist_ok=True)
            vis = np.hstack([square_resized, templates[best_match]])
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            cv2.putText(vis_bgr, f"{best_match} ({max_val:.2f})", (5, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imwrite(f'./debug_frames/match/{img_name}_match.png', vis_bgr)
        return best_match
    return None

def warp_board(crop, points):
    """Warp the chessboard to a fixed 552x552 top-down view (8x8 squares of 69x69)."""
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
    """Split the warped board into 64 squares of fixed 69x69 size with padding."""
    os.makedirs(debug_dir, exist_ok=True)
    squares = []
    square_names = []
    height, width = board_img.shape[:2]
    dy, dx = 69, 69  # Fixed square size

    # Validate board size and add padding if needed
    if height < 8 * dy or width < 8 * dx:
        logging.warning(f"Warped board too small for 69x69 squares: {width}x{height}, need at least {8*dx}x{8*dy}")
        # Pad with black if necessary (simple approach)
        top_pad = max(0, 8 * dy - height)
        left_pad = max(0, 8 * dx - width)
        if top_pad > 0 or left_pad > 0:
            board_img = cv2.copyMakeBorder(board_img, 0, top_pad, 0, left_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            height, width = board_img.shape[:2]
            logging.info(f"Padded board to {width}x{height}")

    for row in range(8):
        for col in range(8):
            y_start, y_end = row * dy, (row + 1) * dy
            x_start, x_end = col * dx, (col + 1) * dx
            square = board_img[y_start:y_end, x_start:x_end]
            if square.size == 0 or square.shape[0] < dy or square.shape[1] < dx:
                logging.warning(f"Empty or undersized square at row {row}, col {col}")
                continue
            gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if len(square.shape) == 3 else square
            #mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            #                            cv2.THRESH_BINARY_INV, 15, 5)
            #white_bg = np.full_like(gray, 255)
            #piece_on_white = np.where(mask == 255, gray, white_bg)
            resized = cv2.resize(gray, (68, 68))
            name = f'square_r{row}_c{col}.png'
            cv2.imwrite(os.path.join(debug_dir, name), resized)
            squares.append(resized)
            square_names.append(name)
    return squares, square_names

def generate_fen(squares, square_names, templates, debug=False):
    """Generate FEN string from detected pieces."""
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

    fen = '/'.join(fen_rows) + ' w KQkq - 0 1'  # Simplified FEN
    return fen

def main():
    image_path = "./ChessBoard_Test.png"
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Cannot load image at {image_path}")
        return
    logging.info(f"Input image shape: {image.shape}")

    try:
        # Instead of auto-detecting, use fixed points
        fixed_points = np.array([
            [1221, 247],  # Top-left
            [1774, 247],  # Top-right
            [1774, 803],  # Bottom-right
            [1221, 803]   # Bottom-left
        ], dtype="float32")

        crop = image  # Use full image since we now use fixed coordinates
        logging.info(f"Warping board with fixed points: {fixed_points.tolist()}")
        board = warp_board(crop, fixed_points)
        logging.info(f"Warped board shape: {board.shape}")
        os.makedirs('./debug_frames', exist_ok=True)
        cv2.imwrite('./debug_frames/warped_board.png', board)

        squares, square_names = split_into_squares(board)
        templates = load_templates("templates", debug_dir="debug_output", debug=True)
        fen = generate_fen(squares, square_names, templates, debug=True)
        logging.info(f"Generated FEN: {fen}")

        # Save final board visualization
        cv2.imwrite('./debug_frames/final_board.png', board)

    except ValueError as e:
        logging.error(f"Processing error: {e}")


if __name__ == "__main__":
    main()
