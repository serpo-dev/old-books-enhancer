import cv2
import numpy as np
import os


def get_skew_angle_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        for rho, theta in line:
            angle = np.degrees(theta) - 90
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    print(f"[INFO] Найден медианный угол через Хаф: {median_angle:.2f}°")
    return round(median_angle, 2)


def deskew_by_hough(image):
    angle = get_skew_angle_hough(image)
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def find_page_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = image.shape[:2]
    page_contour = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_w * img_h * 0.1:  # игнорируем мелкие объекты (например, печати)
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 or len(approx) >= 4:
            if area > max_area:
                max_area = area
                page_contour = approx

    if page_contour is not None:
        x, y, w, h = cv2.boundingRect(page_contour)
        padding = int(min(w, h) * 0.02)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_w - x, w + padding * 2)
        h = min(img_h - y, h + padding * 2)
        return x, y, w, h

    return 0, 0, img_w, img_h


def remove_borders(image, threshold=10, margin=0.02):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    left = 0
    while left < w and np.mean(gray[:, left]) < threshold:
        left += 1
    right = w - 1
    while right > 0 and np.mean(gray[:, right]) < threshold:
        right -= 1
    top = 0
    while top < h and np.mean(gray[top, :]) < threshold:
        top += 1
    bottom = h - 1
    while bottom > 0 and np.mean(gray[bottom, :]) < threshold:
        bottom -= 1

    min_margin = int(max(h, w) * margin)
    left = max(left - 10, min_margin)
    right = min(right + 10, w - min_margin)
    top = max(top - 10, min_margin)
    bottom = min(bottom + 10, h - min_margin)

    cropped = image[top:bottom, left:right]

    # Создаем белый фон
    result = np.full_like(image, fill_value=255)
    ch, cw = cropped.shape[:2]
    x_offset = (w - cw) // 2
    y_offset = (h - ch) // 2
    result[y_offset : y_offset + ch, x_offset : x_offset + cw] = cropped

    return result


def resize_to_a4_ratio(image, a4_ratio=1.414):
    h, w = image.shape[:2]
    current_ratio = h / w

    if abs(current_ratio - a4_ratio) < 0.01:
        return image

    if current_ratio > a4_ratio:
        new_w = int(h / a4_ratio)
        pad = (new_w - w) // 2
        result = cv2.copyMakeBorder(
            image,
            0,
            0,
            pad,
            new_w - w - pad,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )
    else:
        new_h = int(w * a4_ratio)
        pad = (new_h - h) // 2
        result = cv2.copyMakeBorder(
            image,
            pad,
            new_h - h - pad,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

    return result


def process_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    supported_extensions = (".jpg", ".jpeg", ".png")
    files = [
        f for f in os.listdir(input_dir) if f.lower().endswith(supported_extensions)
    ]

    if not files:
        print("[ERROR] В папке нет подходящих файлов для обработки.")
        return

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        print(f"[INFO] Обработка файла: {filename}")

        image = cv2.imread(input_path)
        if image is None:
            print(f"[ERROR] Не удалось загрузить изображение: {input_path}")
            continue

        x, y, w, h = find_page_contour(image)
        roi_image = image[y : y + h, x : x + w]

        corrected = deskew_by_hough(roi_image)
        cleaned = remove_borders(corrected)
        final_image = resize_to_a4_ratio(cleaned)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, final_image)
        print(f"[SUCCESS] Сохранено: {output_path}")


if __name__ == "__main__":
    INPUT_FOLDER = "processed_images"
    OUTPUT_FOLDER = "corrected"

    print("[START] Начинаем обработку изображений...")
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    print("[DONE] Все изображения обработаны.")
