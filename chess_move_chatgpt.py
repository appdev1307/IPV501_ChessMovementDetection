import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

def order_points(pts):
    #if len(pts) != 4:
    #    raise ValueError("Exactly 4 points are required")
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
    crop = image[0:h, 0:w]  # Use full image
    if crop.size == 0:
        raise ValueError("Cropped image is empty")
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpened = cv2.filter2D(gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    blurred = cv2.GaussianBlur(sharpened, (5, 5), 0)
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
            # if len(points) == 4: to be checked why it has more than 4
            return crop, points                        
    
    print("No valid board contour found")
    return crop, None

def load_templates(template_dir="templates"):
    pieces = ["P", "N", "B", "R", "Q", "K", "pb", "nb", "bb", "rb", "qb", "kb"]
    templates = {}
    os.makedirs('./debug_frames', exist_ok=True)

    for p in pieces:
        img_path = os.path.join(template_dir, f"{p}.png")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load with alpha if present

        if img is None:
            raise ValueError(f"Template for {p} not found at {img_path}")

        # If image has alpha channel (RGBA), use it to mask the piece
        if img.shape[2] == 4:
            bgr, alpha = img[:, :, :3], img[:, :, 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Find the largest contour – assumed to be the piece
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError(f"No contour found for {p}")
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Crop to the piece region and resize
        piece_crop = gray[y:y+h, x:x+w]
        piece_resized = cv2.resize(piece_crop, (68, 68))

        # Final thresholding to make it binary
        _, final_template = cv2.threshold(piece_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        templates[p] = final_template

        # Optional debug output
        cv2.imwrite(f'./debug_frames/template_clean_{p}.png', final_template)
        print(f"✔ Processed template: {p}")

    return templates



def match_piece(square_img, img_name, templates, threshold=0.6):
    if square_img.size == 0:
        print("Warning: Empty square image.")
        return None

    square_gray = square_img if len(square_img.shape) == 2 else cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
    square_resized = cv2.resize(square_gray, (68, 68))
    
    # Preprocess with thresholding to reduce background influence
    _, square_thresh = cv2.threshold(square_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    max_val = 0
    best_match = None
    for piece, template in templates.items():
        try:
            res = cv2.matchTemplate(square_thresh, template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > max_val:
                max_val = val
                best_match = piece
                # Save debug match image
                debug_vis = cv2.hconcat([square_thresh, template])
                cv2.imwrite(f'./debug_frames/match_{img_name}_{piece}_{val:.2f}.png', debug_vis)
        except cv2.error as e:
            print(f"Error matching {piece}: {e}")

    if max_val >= threshold:
        print(f"{img_name} matched: {best_match}, confidence={max_val:.3f}")
        return best_match
    else:
        print(f"{img_name} no match. Best: {best_match}, confidence={max_val:.3f}")
        return None

def warp_board(crop, points):
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 8 or height < 8:
        raise ValueError(f"Warped image too small: {width}x{height}")
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(crop, M, (width, height))
    if warped.size == 0 or warped.shape[0] == 0 or warped.shape[1] == 0:
        raise ValueError("Warped image is empty or invalid")
    return warped

def split_into_squares(board_img):
    squares = []
    square_names = []
    height, width = board_img.shape[:2]
    #dy, dx = height // 8, width // 8
    dy, dx = 69, 69
    for row in range(8):
        for col in range(8):
            square = board_img[row*dy:(row+1)*dy, col*dx:(col+1)*dx]
            squares.append(square)
            square_names.append(f'square_r{row}_c{col}.png')
            cv2.imwrite(f'./debug_frames/square_r{row}_c{col}.png', square)
    return squares, square_names

def generate_fen(squares, square_names, templates):
    fen_rows = []
    i = 0
    for piece in squares:
        #print(f'squares  {square_names[i]}')
        piece = match_piece(piece, square_names[i], templates)
        i = i + 1
    
    return '/'.join(fen_rows)

def main():
    image_path = "./ChessBoard_Test.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return
    print(f"Input image shape: {image.shape}")
    
    try:
        crop, points = extract_digital_board(image, debug=True)
        if points is None:
            print("❌ Digital chess board not detected.")
            return
        
        print(f"Warping board with points: {points}")
        board = warp_board(crop, points)
        print(f"Warped board shape: {board.shape}")
        os.makedirs('./debug_frames', exist_ok=True)
        cv2.imwrite('./debug_frames/warped_board.png', board)
        
        squares, square_names = split_into_squares(board)
        templates = load_templates("templates")
        fen = generate_fen(squares, square_names,templates)
        print("Generated FEN:", fen)
        
        plt.ion()
        plt.imshow(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
        plt.title("Top-down Digital Chess Board")
        plt.axis("off")
        plt.savefig('./debug_frames/FEN.png')
        plt.close()
    except ValueError as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main()
