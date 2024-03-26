from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# Single list of sentences - Possible tens of thousands of sentences
sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
]

output_txt_paths = [
    'articles_text/1intro.txt',
    'articles_text/2relation.txt',
    'articles_text/3count.txt',
    'articles_text/4probability.txt',
    'articles_text/libya.txt',
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


documents = []

for path in output_txt_paths:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
            documents.append(text)
    except FileNotFoundError:
        print(f"File {path} not found. Skipping.")

paraphrases = util.paraphrase_mining(model, documents)

for paraphrase in paraphrases[0:10]:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(output_txt_paths[i], output_txt_paths[j], score))