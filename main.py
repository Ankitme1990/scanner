from pdf_utils import convert_pdf_to_images, extract_tables_from_pages
#from image_utils import process_image
from text_utils import extract_text_from_image, write_text_to_file
from table_utils import write_tables_to_file
from image_utils2 import process_image

# Path to the PDF file
pdf_path = "Cluster Mining Plan.pdf"

# Number of pages in the PDF
num_pages = 10

# Path to the Ghostscript executable
ghostscript_path = r"C:\Program Files\gs\gs10.00.0\bin\gswin64.exe"

# Convert PDF pages to images
convert_pdf_to_images(pdf_path, num_pages, ghostscript_path)

# Extract text from images and combine into a single text file
combined_text = ""
for page_num in range(num_pages):
    image_path = f"page_{page_num}.png"
    image = process_image(image_path)
    text = extract_text_from_image(image)
    combined_text += text

write_text_to_file(combined_text, "combined_text.txt")

# Extract tables from PDF pages and write to a file
extracted_tables = extract_tables_from_pages(pdf_path, num_pages)
write_tables_to_file(extracted_tables, "extracted_tables.csv")
