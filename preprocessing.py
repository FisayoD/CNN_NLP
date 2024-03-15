import fitz  # PyMuPDF
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def preprocess_text(text):
    """Remove commas, full stops, and other punctuation."""
    punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punctuation = "".join([char for char in text if char not in punctuation])
    return no_punctuation

def read_pdf(file_path):
    """Read and preprocess text from a PDF file."""
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return preprocess_text(full_text)

def write_text_to_pdf(output_pdf_path, input_text):
    """Write preprocessed text to a new PDF file, handling long texts."""
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    text_object = c.beginText(40, 750)  
    text_object.setFont("Helvetica", 12)
    
    for line in input_text.splitlines(): 
        text_object.textLine(line)
    c.drawText(text_object)  
    c.save()


def write_text_to_file(output_file_path, text):
    """Write preprocessed text to a text file."""
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(text)


 
input_pdf_paths = [
    'articles/1intro.pdf',
    'articles/2relation.pdf',
    'articles/3count.pdf',
    'articles/4probability.pdf',
    'articles/libya.pdf',
    'articles/Homework_3_Fisayo_Ojo.pdf',
    'articles/An Investigation of the Pattern And Environmental Impact of Oil.pdf', 
    'articles/Analysis of oil spill impacts along pipelines.pdf',  
    'articles/Causes and Terrain of Oil Spillage in Niger Delta.pdf', 
    'articles/Deficient legislation sanctioning oil spill.pdf',  
    'articles/Effects of Oil Spillage (Pollution) on Agricultural Production in Delta.pdf', 
    'articles/Effects of oil spills on fish production in the Niger Delta.pdf',  
    'articles/EFFECTS OF OIL SPILLAGE ON FISH IN NIGERIA.pdf',  
    'articles/Effects of oil spills on fish production in the Niger Delta.pdf',  
    'articles/Environmental Consequences of Oil Spills on Marine Habitats and the Mitigating Measures—The Niger Delta Perspective.pdf',
    'articles/ENVIRONMENTAL IMPACTS OF OIL EXPLORATION.pdf',  
    'articles/Evaluation of the Impacts of Oil Spill Disaster on Communities in Niger Delta, Nigeria.pdf',  
    'articles/Impacts and Management of Oil Spill Pollution along the Nigerian Coastal Areas.pdf', 
    'articles/Impacts of Oil Exploration (Oil and Gas Conflicts; Niger Delta as a Case Study).pdf', 
    'articles/Impacts of Oil Production on Nigeria‘s Waters.pdf',  
    'articles/NIGERIA OIL POLLUTION, POLITICS AND POLICY.pdf', 
    'articles/Oil Pollution in Nigeria and the Issue of Human Rights of the Victims.pdf', 
    'articles/Oil Spills and Human Health.pdf',  
    'articles/OIL SPILLS IN THE NIGER DELTA.pdf', 
    'articles/mining.pdf',
    'articles/espionage.pdf',
    'articles/Press Coverage of Environmental Pollution In The Niger Delta Region of Nigeria.pdf',  
    'articles/Shell will sell big piece of its Nigeria oil business, but activists want pollution cleaned up first _ AP News.pdf'  
]

output_txt_paths = [
    'articles_text/1intro.txt',
    'articles_text/2relation.txt',
    'articles_text/3count.txt',
    'articles_text/4probability.txt',
    'articles_text/libya.txt',
    'articles_text/mining.txt',
    'articles_text/espionage.txt',
    'articles_text/Homework_3_Fisayo_Ojo.txt',
    'articles_text/An Investigation of the Pattern And Environmental Impact of Oil.txt',
    'articles_text/Analysis of oil spill impacts along pipelines..txt',
    'articles_text/Causes and Terrain of Oil Spillage in Niger Delta.txt',
    'articles_text/Deficient legislation sanctioning oil spill.txt',
    'articles_text/Effects of Oil Spillage (Pollution) on Agricultural Production in Delta.txt',  
    'articles_text/Effects of oil spills on fish production in the Niger Delta.txt',  
    'articles_text/EFFECTS OF OIL SPILLAGE ON FISH IN NIGERIA.txt', 
    'articles_text/Effects of oil spills on fish production in the Niger Delta.txt',  
    'articles_text/Environmental Consequences of Oil Spills on Marine Habitats and the Mitigating Measures—The Niger Delta Perspective.txt',  
    'articles_text/ENVIRONMENTAL IMPACTS OF OIL EXPLORATION.txt', 
    'articles_text/Evaluation of the Impacts of Oil Spill Disaster on Communities in Niger Delta, Nigeria.txt',  
    'articles_text/Impacts and Management of Oil Spill Pollution along the Nigerian Coastal Areas.txt',  
    'articles_text/Impacts of Oil Exploration (Oil and Gas Conflicts; Niger Delta as a Case Study).txt',  
    'articles_text/Impacts of Oil Production on Nigeria‘s Waters.txt', 
    'articles_text/NIGERIA OIL POLLUTION, POLITICS AND POLICY.txt',  
    'articles_text/Oil Pollution in Nigeria and the Issue of Human Rights of the Victims.txt', 
    'articles_text/Oil Spills and Human Health.txt',  
    'articles_text/OIL SPILLS IN THE NIGER DELTA.txt',  
    'articles_text/Press Coverage of Environmental Pollution In The Niger Delta Region of Nigeria.txt',  
    'articles_text/Shell will sell big piece of its Nigeria oil business, but activists want pollution cleaned up first _ AP News.txt'  
]



for input_path, output_txt_path in zip(input_pdf_paths, output_txt_paths):
    preprocessed_text = read_pdf(input_path)
    write_text_to_file(output_txt_path, preprocessed_text)