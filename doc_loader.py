import csv
from PyPDF2 import PdfReader

#Loading CSV data for Input to model

def load_csv(file_path):
    requirements = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            requirements.append(row)
    return requirements

#Loading PDF data for knowledge base
def load_pdf(pdf_path):
    # Create a PdfReader object
    reader = PdfReader(pdf_path)

    # Initialize a variable to store the text
    pdf_text = ""

    # Iterate through all pages and extract text
    for page in reader.pages:
        pdf_text += page.extract_text()
        # Print the extracted text
        # print(pdf_text)

    return pdf_text

def get_req_instance():
    csv_file_path = "Req EPS/SystemReq_EPS.csv"

    requirements_csv = load_csv(csv_file_path)
    req_instance = requirements_csv[14]['Primary Text']

    return req_instance

