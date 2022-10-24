from PyPDF2 import PdfFileReader, PdfFileWriter
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
import pytesseract, cv2
import logging

reader = PdfFileReader("../../../Downloads/1.pdf", "rb")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
logging.info(f"Reading {reader.numPages}")

def ocr(image):
    # calling the processor is equivalent to calling the feature extractor
    pixel_values = processor(image, return_tensors="pt").pixel_values
    print(pixel_values.shape)

    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

for page in reader.pages:
    for image in page.images:
        # image: .name, .data
        with open(image.name, "wb") as file:
            file.write(image.data)
        img = cv2.cvtColor(cv2.imread(image.name), cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(img, lang='eng')
        print(text)
        
        # print(ocr(img))