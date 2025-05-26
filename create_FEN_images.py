import cv2
import os

def split_board_into_squares(image):
    h, w = image.shape[:2]
    square_size = h // 8
    squares = []
    for i in range(8):
        for j in range(8):
            x1, y1 = j * square_size, i * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            square = image[y1:y2, x1:x2]
            squares.append(square)
    return squares

def save_square(square, name, folder="templates"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{name}.png")
    square_resized = cv2.resize(square, (50, 50))
    cv2.imwrite(filepath, square_resized)
    print(f"Saved: {filepath}")

def extract_templates(image_path):
    image = cv2.imread(image_path)
    squares = split_board_into_squares(image)

    for idx, square in enumerate(squares):
        cv2.imshow("Square", square)
        print(f"[{idx}] Enter label (K/Q/R/B/N/P or k/q/r/b/n/p), or leave empty to skip:")
        key = input("Label: ").strip()
        if key in ['K','Q','R','B','N','P','k','q','r','b','n','p']:
            save_square(square, key)
        else:
            print("Skipped.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_templates("board_topdown.png")
