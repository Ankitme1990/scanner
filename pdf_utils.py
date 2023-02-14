import subprocess
import tabula
import os

def convert_pdf_to_images(pdf_path, num_pages, ghostscript_path):
    # Loop over all the pages
    for page_num in range(num_pages):
        # Command to convert the PDF page to an image using Ghostscript
        convert_cmd = [ghostscript_path, "-sDEVICE=pngalpha", "-dFirstPage={}".format(page_num + 1),
                       "-dLastPage={}".format(page_num + 1), "-dNOPAUSE", "-dBATCH", "-r300",
                       "-sOutputFile=page_{}.png".format(page_num), pdf_path]

        # Convert the PDF page to an image using Ghostscript
        subprocess.run(convert_cmd, check=True)

def extract_tables_from_pages(pdf_path, num_pages):
    extracted_tables = []
    # Loop over all the pages
    for page_num in range(num_pages):
        # Extract tables using Tabula
        tables = tabula.read_pdf(pdf_path, pages=str(page_num+1), multiple_tables=True)

        # Append the extracted tables to the list
        for table in tables:
            extracted_tables.append(table)
    return extracted_tables
