import cv2
import numpy as np
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

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
    #crop = image[0:h, int(w * 0.4):w]  # Wider right crop
    crop = image[0:h, 0:w]  # Wider right crop

    # Preprocessing
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
        plt.savefig('./debug_frames/_dplot.png')
        #plt.show()

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) >= 4 and cv2.contourArea(c) > 5000:
            return crop, approx.reshape(-1, 2)

    return crop, None


def warp_board(crop, points):
    rect = order_points(points)
    (tl, tr, br, bl) = rect

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(crop, M, (width, height))
    return warped

def split_into_squares(board_img):
    squares = []
    height, width = board_img.shape[:2]
    dy, dx = height // 8, width // 8
    for row in range(8):
        for col in range(8):
            square = board_img[row*dy:(row+1)*dy, col*dx:(col+1)*dx]
            squares.append(square)
    return squares

def classify_piece(square_img):
    # Stub: Replace with model/template-based classifier
    return '1'

def generate_fen(squares):
    fen = ""
    for i in range(8):
        empty_count = 0
        for j in range(8):
            symbol = classify_piece(squares[i * 8 + j])
            if symbol == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen += str(empty_count)
                    empty_count = 0
                fen += symbol
        if empty_count > 0:
            fen += str(empty_count)
        if i < 7:
            fen += '/'
    return fen + " w KQkq - 0 1"

def main():
    image_path = "./ChessBoard_Test.png"  # Replace with your file
    image = cv2.imread(image_path)

    print(f"Read image at {image_path}")

    if image is None:
        print(f"Error: Cannot load image at {image_path}")
        return

    crop, points = extract_digital_board(image, debug=True)  # Set debug=True

    if points is not None:
        board = warp_board(crop, points)
        squares = split_into_squares(board)
        fen = generate_fen(squares)

        print("Generated FEN:", fen)
        plt.ion()
        plt.imshow(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
        plt.title("Top-down Digital Chess Board")
        plt.axis("off")
        #plt.show()
        plt.savefig('./debug_frames/FEN.png')
    else:
        print("❌ Digital chess board not detected.")

if __name__ == "__main__":
    main()
