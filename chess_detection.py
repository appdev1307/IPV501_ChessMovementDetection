import cv2
import numpy as np
import os

# Đọc ảnh và template
image = cv2.imread('chessboard_test_image_1.PNG')
template = cv2.imread('chessboard_template.PNG')
if image is None or template is None:
    print("Không thể đọc ảnh hoặc template!")
    exit()

# Chuyển ảnh sang không gian màu HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Điều chỉnh phạm vi màu của bàn cờ (xanh lá và trắng) - Mở rộng hơn
lower_green = np.array([15, 10, 10])  # Mở rộng để bao quát sắc xanh
upper_green = np.array([120, 255, 255])
lower_white = np.array([0, 0, 100])   # Bao gồm trắng nhạt hơn
upper_white = np.array([180, 70, 255])

# Tạo mask cho màu xanh lá và trắng
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_white = cv2.inRange(hsv, lower_white, upper_white)
mask = cv2.bitwise_or(mask_green, mask_white)

# Lưu mask để debug
cv2.imwrite('mask_green_debug.jpg', mask_green)
cv2.imwrite('mask_white_debug.jpg', mask_white)
cv2.imwrite('mask_combined_debug.jpg', mask)

# Loại bỏ nhiễu bằng morphologic operations
kernel = np.ones((5, 5), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)
mask = cv2.erode(mask, kernel, iterations=2)

# Lưu mask sau morphologic
cv2.imwrite('mask_morph_debug.jpg', mask)

# Áp dụng mask lên ảnh gốc để giảm nhiễu cho template matching
masked_image = cv2.bitwise_and(image, image, mask=mask)
gray_masked = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# Chuyển template sang grayscale
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Resize template để khớp với kích thước bàn cờ trong ảnh (ước lượng)
scale_factor = 0.5  # Điều chỉnh scale_factor nếu cần
resized_template = cv2.resize(gray_template, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

# Khớp template
result = cv2.matchTemplate(gray_masked, resized_template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Kiểm tra độ tin cậy của matching
if max_val < 0.4:  # Ngưỡng độ tin cậy (có thể điều chỉnh)
    print("Template matching không đủ độ tin cậy! Thử điều chỉnh scale_factor hoặc ngưỡng màu HSV.")
    exit()

# Lấy kích thước template đã resize
h, w = resized_template.shape

# Tọa độ vùng bàn cờ
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# Cắt vùng bàn cờ
board = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# Lưu bàn cờ đã cô lập để kiểm tra
output_dir = 'chessboard_extracted'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cv2.imwrite(os.path.join(output_dir, 'extracted_board.jpg'), board)

# Chia lưới 8x8 dựa trên kích thước bàn cờ đã cắt
board_height, board_width, _ = board.shape
square_size_x = board_width // 8
square_size_y = board_height // 8

# Tạo thư mục lưu các ô
squares_dir = 'chessboard_squares'
if not os.path.exists(squares_dir):
    os.makedirs(squares_dir)

# Duyệt qua các ô và phân tích
for i in range(8):
    for j in range(8):
        # Tọa độ ô
        x1 = j * square_size_x
        x2 = (j + 1) * square_size_x
        y1 = i * square_size_y
        y2 = (i + 1) * square_size_y
        
        # Cắt ô
        square = board[y1:y2, x1:x2]
        gray_square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
        
        # Xác định ô trắng hay đen (dựa trên trung bình intensity)
        avg_intensity = np.mean(gray_square)
        color = "white" if avg_intensity > 127 else "black"
        
        # Phát hiện quân cờ bằng cách đếm cạnh
        edges = cv2.Canny(gray_square, 50, 150)
        edge_count = np.sum(edges > 0)
        has_piece = edge_count > 200  # Ngưỡng có thể điều chỉnh
        
        # Lưu ô
        label = f"{color}_piece" if has_piece else f"{color}_empty"
        cv2.imwrite(os.path.join(squares_dir, f'square_{i}_{j}_{label}.jpg'), square)

print(f"Bàn cờ đã được cô lập và lưu tại: {output_dir}/extracted_board.jpg")
print(f"Các ô đã được lưu trong thư mục: {squares_dir}")