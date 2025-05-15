from PIL import Image, ImageOps, ImageEnhance
import fitz  # PyMuPDF
import os
from io import BytesIO

def process_image(image):
    gray_image = ImageOps.grayscale(image)
    adjusted_image = ImageOps.autocontrast(gray_image, cutoff=25 * 0.1)
    enhancer = ImageEnhance.Brightness(adjusted_image)
    return enhancer.enhance(1.2)

def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        
        # Загружаем изображение через PIL корректно
        img = Image.open(BytesIO(img_data)).convert("RGB")
        
        processed_img = process_image(img)
        output_path = os.path.join(output_folder, f"page_{i+1:03d}.png")
        processed_img.save(output_path)
        print(f"[SUCCESS] Обработана страница {i+1} и сохранена как {output_path}")
    doc.close()

if __name__ == "__main__":
    input_pdf = "input.pdf"
    output_dir = "processed_images"

    if not os.path.exists(input_pdf):
        print(f"[ERROR] Файл '{input_pdf}' не найден.")
    else:
        convert_pdf_to_images(input_pdf, output_dir, dpi=300)