import pandas as pd

s08 = pd.read_csv("S08_question_answer_pairs.txt",sep="\t")
s09 = pd.read_csv("S09_question_answer_pairs.txt",sep="\t")
s10 = pd.read_csv("S10_question_answer_pairs.txt",sep="\t")
mini_wiki = pd.concat([s08,s09,s10])
mini_wiki = mini_wiki.drop_duplicates(subset=["Question","ArticleFile"])
mini_wiki["ArticleTitle"] = mini_wiki["ArticleTitle"].apply(lambda title: " ".join([word.capitalize() for word in title.strip().split("_")]))
mini_wiki = mini_wiki.dropna(subset="Answer")
mini_wiki = mini_wiki.sort_values(by="ArticleTitle").reset_index(drop=True) 

mini_wiki.to_csv("backend/mini_wiki.csv",index=False)