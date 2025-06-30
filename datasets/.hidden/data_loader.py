import json
import pandas as pd
import os

class DataLoader:
    def load_dataset_winobias():
        files = ['anti_stereotyped_type1.txt.dev', 'anti_stereotyped_type1.txt.test', 'anti_stereotyped_type2.txt.dev', 'anti_stereotyped_type2.txt.test',
            'pro_stereotyped_type1.txt.dev', 'pro_stereotyped_type1.txt.test', 'pro_stereotyped_type2.txt.dev', 'pro_stereotyped_type2.txt.test']

        sentences = []

        for file in files:
            file_path = f'./data/WinoBias/data/{file}'
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences += f.readlines()

        return sentences
            
    def load_dataset_winobiasplus():
        sentences = []
        file_path = f'./data/WinoBias+/data/WinoBias+.txt'
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences += f.readlines()

        return sentences
        
    def load_dataset_winogender():
        df = pd.read_csv('./data/WinoGender/data/all_sentences.tsv', sep='\t')
        sentences = df['sentence'].tolist()

        return sentences
        
    def load_dataset_gap():
        files = ['gap-development.tsv', 'gap-test.tsv', 'gap-validation.tsv']

        sentences = []

        for file in files:
            df = pd.read_csv(f'./data/GAP/data/{file}', sep='\t')

            sentence = df['Text'].tolist()
            sentences += sentence

        return sentences

    def load_dataset_bug():
        files = ['balanced_BUG.csv', 'full_BUG.csv', 'gold_BUG.csv']

        sentences = []

        for file in files:
            df = pd.read_csv(f'./data/BUG/data/{file}')

            sentence = df['sentence_text'].tolist()
            sentences += sentence
        
        return sentences
    
    def load_dataset_stereset():
        files = ["./data/StereoSet/data/dev.json", "./data/StereoSet/data/test.json"]
        all_sentences = []

        for file in files:
            with open(file, 'r') as file:
                data = json.load(file)

                for key in ['intersentence', 'intrasentence']:
                    if key in data['data']:
                        for entry in data['data'][key]:
                            context = entry.get('context', "")
                            if context:
                                all_sentences.append(context)
                            
                            # Extract sentences from each entry
                            sentences = entry.get('sentences', [])
                            for sentence_data in sentences:
                                sentence = sentence_data.get('sentence', "")
                                if sentence:
                                    all_sentences.append(sentence)

        return all_sentences

    def load_dataset_becpro():
        df = pd.read_csv(f'./data/BEC-Pro/data/BEC-Pro_EN.tsv', sep='\t')
        sentences = df['Sentence'].tolist()

        return sentences

    def load_dataset_honest():
        files = ['binary/en_template.tsv', 'queer_nonqueer/en_template.tsv']

        sentences = []

        for file in files:
            df = pd.read_csv(f'./data/HONEST/data/{file}', sep='\t')

            sentence = df['template_masked'].tolist()
            sentences += sentence

        return sentences

    def load_dataset_crowspairs():
        sentences = []
        df = pd.read_csv(f'./data/CrowS-Pairs/data/crows_pairs_anonymized.csv')

        sentence = df['sent_more'].tolist()
        sentences += sentence
        sentence = df['sent_less'].tolist()
        sentences += sentence

        return sentences

    def load_dataset_eec():
        df = pd.read_csv(f'./data/EEC/data/Equity-Evaluation-Corpus.csv')
        sentences = df['Sentence'].tolist()

        return sentences

    def load_dataset_panda():
        directory_path = "./data/PANDA/data"  

        json_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".jsonl"):
                    json_files.append(os.path.join(root, file))

        sentences = []

        for file in json_files:
            with open(file, 'r', encoding='utf-8') as file:
                for line in file:
                    # Parse each line as a JSON object
                    data = json.loads(line)
                        
                    # Extract 'original' and 'rewrite' fields
                    original = data.get('original', '')
                    rewrite = data.get('rewrite', '')
                        
                    # Add the extracted fields to the list as a tuple
                    sentences.append(original)
                    sentences.append(rewrite)
            
            return sentences

    def load_dataset_redditbias():
        directory_path = "./data/RedditBias/data"  

        json_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))

        sentences = []
        for file in json_files:
            df = pd.read_csv(file)

            sentence = df['comments_processed'].tolist()
            sentences += sentence

        return sentences

    def load_dataset_winoqueer():
        sentences = []
        df = pd.read_csv(f'./data/WinoQueer/data/winoqueer_final.csv')
        sentence = df['sent_x'].tolist()
        sentences += sentence
        sentence = df['sent_y'].tolist()
        sentences += sentence

        return sentences

    def load_dataset_fairprism():
        files = ['fairprism_aggregated.csv', 'fairprism_disaggregated.csv']

        sentences = []

        for file in files:
            df = pd.read_csv(f'./data/FairPrism/data/{file}')

            sentence = df['Human Input'].tolist()
            sentences += sentence
            sentence = df['AI Output'].tolist()
            sentences += sentence
        
        return sentences

    def load_dataset_bold():
        files = ['gender_prompt.json',
                'political_ideology_prompt.json',
                'profession_prompt.json',
                'race_prompt.json',
                'religious_ideology_prompt.json']

        sentences = []

        for file in files:
            with open(f'./data/BOLD/data/prompts/{file}', 'r') as file:
                data = json.load(file)

                # Extract sentences from all categories in the dataset
                cur_sentences = [sentence for category in data.values() for actor_info in category.values() for sentence in actor_info]
                sentences += cur_sentences
        
        return sentences

    def load_dataset_holisticbias():
        df = pd.read_csv(f'./data/HolisticBias/data/data/sentences.csv')

        sentences = df['text'].tolist()

        return sentences

    def load_dataset_bbq():
        directory_path = "./data/BBQ/data"  

        json_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".jsonl"):
                    json_files.append(os.path.join(root, file))

        sentences = []

        for file in json_files:
            with open(file, 'r', encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                        
                    context = data.get('context', '')
                    question = data.get('question', '')
                        
                    sentences.append(context)
                    sentences.append(question)
        return sentences

    def load_dataset_unqover():
        directory_path = "./data/UnQover/data"  

        json_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))

        sentences = []

        for file in json_files:
            with open(file, 'r', encoding='utf-8') as f:

                    data = json.load(f)
                        
                    for key, value in data.items():
                        context = value.get("context", "")
                        if "--" in context:
                            context_before = context.split("--")[0].strip()  
                            sentences.append(context_before)

                        for q_key in value:
                            if q_key.startswith("q"):  
                                question_data = value[q_key]
                                if "question" in question_data:
                                    sentences.append(question_data["question"])
        return sentences

    def load_dataset_ceb():
        directory_path = "./data/CEB/data"  

        json_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))

        sentences = []
        for file in json_files:
            with open(f'{file}', 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                        
                    for prompt_data in data:
                        sentences.append(prompt_data["prompt"])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {file}")

        return sentences

    

    