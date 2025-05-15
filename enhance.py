from PIL import Image, ImageOps, ImageEnhance
import fitz  # PyMuPDF
import os
from io import BytesIO

def process_image(image):
    gray_image = ImageOps.grayscale(image)
    adjusted_image = ImageOps.autocontrast(gray_image, cutoff=25 * 0.1)
    enhancer = ImageEnhance.Brightness(adjusted_image)
    return enhancer.enhance(1.2)

def split_image_horizontally(image):
    width, height = image.size
    left = image.crop((0, 0, width // 2, height))
    right = image.crop((width // 2, 0, width, height))
    return left, right

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    doc = fitz.open(pdf_path)
    page_counter = 0

    for page_num, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data)).convert("RGB")

        width, height = img.size

        if width > height:
            print(f"[INFO] Страница {page_num + 1} альбомная — разделяем на две.")
            left, right = split_image_horizontally(img)
            processed_left = process_image(left)
            processed_right = process_image(right)

            left_path = os.path.join(output_folder, f"page_{page_counter + 1:03d}.png")
            right_path = os.path.join(output_folder, f"page_{page_counter + 2:03d}.png")

            processed_left.save(left_path)
            processed_right.save(right_path)

            page_counter += 2
            print(f"[SUCCESS] Обработаны и сохранены страницы: {left_path}, {right_path}")
        else:
            print(f"[INFO] Страница {page_num + 1} портретная — без изменений.")
            processed_img = process_image(img)
            output_path = os.path.join(output_folder, f"page_{page_counter + 1:03d}.png")
            processed_img.save(output_path)
            page_counter += 1
            print(f"[SUCCESS] Обработана и сохранена страница: {output_path}")

    doc.close()

if __name__ == "__main__":
    input_pdf = "input.pdf"
    output_dir = "processed_images"

    if not os.path.exists(input_pdf):
        print(f"[ERROR] Файл '{input_pdf}' не найден.")
    else:
        convert_pdf_to_images(input_pdf, output_dir, dpi=300)