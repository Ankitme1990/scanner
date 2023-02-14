import tabula
import pandas as pd

def extract_tables(pdf_path, page_num):
    tables = tabula.read_pdf(pdf_path, pages=str(page_num+1), multiple_tables=True)
    return tables

def write_tables_to_csv(tables, csv_path):
    for i, table in enumerate(tables):
        table.to_csv(f"{csv_path}/table_{i+1}.csv", index=False)

def write_tables_to_file(tables, filename):
    with open(filename, "w") as f:
        for i, table in enumerate(tables):
            f.write(f"Table {i+1}:\n{table.to_string(index=False)}\n\n")

def extract_tables_to_csv(pdf_path, num_pages, csv_path):
    for page_num in range(num_pages):
        tables = extract_tables(pdf_path, page_num)
        write_tables_to_csv(tables, csv_path)
