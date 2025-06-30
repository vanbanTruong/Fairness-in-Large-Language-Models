import tools as tools
from data_loader import DataLoader
import json
import sys

def print_invalid_command():
    print("Invalid command!")

command = 1
dataset = []
keywords_dict = {}

with open('./keywords.json') as f:
    keywords_dict = json.load(f)

if sys.argv[1] == "visualize":
    command = 0
elif sys.argv[1] == "keywords":
    command = 1
else:
    print_invalid_command()
    exit()

if sys.argv[2] == "winobias":
    dataset = DataLoader.load_dataset_winobias()
elif sys.argv[2] == "winobias+":
    dataset = DataLoader.load_dataset_winobiasplus()
elif sys.argv[2] == "winogender":
    dataset = DataLoader.load_dataset_winogender()
elif sys.argv[2] == "gap":
    dataset = DataLoader.load_dataset_gap()
elif sys.argv[2] == "bug":
    dataset = DataLoader.load_dataset_bug()
elif sys.argv[2] == "stereoset":
    dataset = DataLoader.load_dataset_stereset()
elif sys.argv[2] == "becpro":
    dataset = DataLoader.load_dataset_becpro()
elif sys.argv[2] == "honest":
    dataset = DataLoader.load_dataset_honest()
elif sys.argv[2] == "crowspairs":
    dataset = DataLoader.load_dataset_crowspairs()
elif sys.argv[2] == "eec":
    dataset = DataLoader.load_dataset_eec()
elif sys.argv[2] == "panda":
    dataset = DataLoader.load_dataset_panda()
elif sys.argv[2] == "redditbias":
    dataset = DataLoader.load_dataset_redditbias()
elif sys.argv[2] == "winoqueer":
    dataset = DataLoader.load_dataset_winoqueer()
elif sys.argv[2] == "bold":
    dataset = DataLoader.load_dataset_bold()
elif sys.argv[2] == "holisticbias":
    dataset = DataLoader.load_dataset_holisticbias()
elif sys.argv[2] == "bbq":
    dataset = DataLoader.load_dataset_bbq()
elif sys.argv[2] == "unqover":
    dataset = DataLoader.load_dataset_unqover()
elif sys.argv[2] == "ceb":
    dataset = DataLoader.load_dataset_ceb()
else:
    print_invalid_command()
    exit()

keywords = keywords_dict[sys.argv[2]]
if command:
    top_k = 100
    if len(sys.argv) > 2:
        top_k = int(sys.argv[3])    
    top_k_keywords = tools.generate_keywords_from_sentences(dataset, top_k)
    print(f'Top {top_k} keywords of {sys.argv[1]}: ', top_k_keywords)
else:
    if len(keywords) == 0:
        keywords = tools.generate_keywords_from_sentences(dataset, 30)  
    tools.visualize_gender_bias_tfidf(dataset, keywords)