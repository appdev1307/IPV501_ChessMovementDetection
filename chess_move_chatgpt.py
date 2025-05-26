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
    for p in pieces:
        img = cv2.imread(os.path.join(template_dir, f"{p}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Template for {p} not found.")
        print(f"{p}.png loaded")
        templates[p] = cv2.resize(img, (68, 68))
    return templates

def match_piece(square_img, img_name, templates, threshold=0.3):
    if square_img.size == 0 or square_img.shape[0] == 0 or square_img.shape[1] == 0:
        print("Warning: Empty square image passed to match_piece")
        return None

    # Check number of channels
    if len(square_img.shape) == 2 or square_img.shape[2] == 1:
        # Already grayscale
        square_gray = square_img
    else:
        # Convert BGR to grayscale
        square_gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)    
        
    square_resized = cv2.resize(square_gray, (68, 68))
    max_val = 0
    best_match = None
    for piece, template in templates.items():
        #print(f"match_piece: template={piece}, shape={template.shape}, "
        #      f"channels={1 if len(template.shape) == 2 else template.shape[2]}, "
        #      f"dtype={template.dtype}")
        
        #if piece == list(templates.keys())[0]:
        #    cv2.imwrite('./debug_frames/template_sample.png', template)
        
        if template.shape[0] > square_resized.shape[0] or template.shape[1] > square_resized.shape[1]:
            print(f"Warning: Template {piece} too large: template={template.shape}, square_resized={square_resized.shape}")
            continue
        if (len(template.shape) == 2) != (len(square_resized.shape) == 2):
            print(f"Warning: Channel mismatch for {piece}: template channels={1 if len(template.shape) == 2 else template.shape[2]}, "
                  f"square_resized channels={1 if len(square_resized.shape) == 2 else square_resized.shape[2]}")
            continue
        if template.dtype != square_resized.dtype:
            print(f"Warning: Dtype mismatch for {piece}: template={template.dtype}, square_resized={square_resized.dtype}")
            continue
        
        try:
            res = cv2.matchTemplate(square_resized, template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > max_val:
                max_val = val
                best_match = piece
                #print(f"matched: template={best_match}, max_val={max_val:.4f}")
        except cv2.error as e:
            print(f"Error in matchTemplate for {piece}: {e}")
            continue
    
    

    if max_val >= threshold:
        print(f"{img_name} matched: template={best_match}, max_val={max_val:.4f}")
        return best_match
    else:
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
