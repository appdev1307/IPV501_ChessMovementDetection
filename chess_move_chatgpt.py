import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#match result
match_result = []

#UNIT Test 1


# Erosion kernel
EROSION_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

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

        resized = cv2.resize(img, (68, 68))
        eroded = cv2.erode(resized, EROSION_KERNEL, iterations=1)
        templates[p] = eroded

        if debug:
            cv2.imwrite(os.path.join(debug_dir, f"{p}_eroded.png"), eroded)
            #logging.info(f"Saved eroded template for {p}")

    return templates

def match_piece(square_img, img_name, templates, threshold=0.6, debug=False):
    if square_img.size == 0 or square_img.shape[0] == 0 or square_img.shape[1] == 0:
        logging.warning(f"Empty square image: {img_name}")
        return None

    square_gray = square_img if len(square_img.shape) == 2 else cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    square_resized = cv2.resize(square_gray, (68, 68))
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

    #logging.info(f"{img_name} match result: best={best_match}, score={max_val:.4f}")
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

def main():
    image_path = "./ChessBoard_Test.png"
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Cannot load image at {image_path}")
        return
    logging.info(f"Input image shape: {image.shape}")

    try:
        fixed_points = np.array([
            [1221, 247],
            [1774, 247],
            [1774, 803],
            [1221, 803]
        ], dtype="float32")

        crop = image
        logging.info(f"Warping board with fixed points: {fixed_points.tolist()}")
        board = warp_board(crop, fixed_points)
        logging.info(f"Warped board shape: {board.shape}")
        os.makedirs('./debug_frames', exist_ok=True)
        cv2.imwrite('./debug_frames/warped_board.png', board)

        squares, square_names = split_into_squares(board)
        templates = load_templates("templates", debug_dir="debug_output", debug=True)
        fen = generate_fen(squares, square_names, templates, debug=True)
        logging.info(f"Generated FEN: {fen}")


        #print(match_result)
        # Unit Test 1
        match_UT1 = []
        match_UT1.append("square_r0_c4.png match =qb")
        match_UT1.append("square_r0_c5.png match =rb")
        match_UT1.append("square_r0_c6.png match =kb")

        match_UT1.append("square_r1_c2.png match =R")
        match_UT1.append("square_r1_c4.png match =bb")
        match_UT1.append("square_r1_c5.png match =pb")
        match_UT1.append("square_r1_c6.png match =pb")
        match_UT1.append("square_r1_c7.png match =pb")

        match_UT1.append("square_r2_c0.png match =bb")
        match_UT1.append("square_r2_c1.png match =pb")

        match_UT1.append("square_r3_c0.png match =pb")
        match_UT1.append("square_r3_c3.png match =rb")

        match_UT1.append("square_r4_c0.png match =P")
        match_UT1.append("square_r4_c4.png match =P")

        match_UT1.append("square_r5_c1.png match =Q")
        match_UT1.append("square_r5_c5.png match =N")
        match_UT1.append("square_r5_c6.png match =P")
        match_UT1.append("square_r5_c7.png match =B")

        match_UT1.append("square_r6_c1.png match =P")
        match_UT1.append("square_r6_c5.png match =P")
        match_UT1.append("square_r6_c7.png match =P")

        match_UT1.append("square_r7_c6.png match =K")

        max_len = max(len(match_result), len(match_UT1))
        for i in range(max_len):
            item1 = match_result[i] if i < len(match_result) else "<missing>"
            item2 = match_UT1[i] if i < len(match_UT1) else "<missing>"
            if item1 != item2:
                print(f"Difference at index {i}: match_result = '{item1}', match_UT1 = '{item2}'")

    except ValueError as e:
        logging.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
