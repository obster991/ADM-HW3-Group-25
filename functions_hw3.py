import requests as r
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import os
from datetime import datetime
import csv
import re
import nltk
from collections import Counter
import numpy as np
import math
import heapq
import time

import re
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
import string
import json

default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')

# Append some recurrent words in synopsis:
frequent_words = ['character','characters','end']
default_stopwords.extend(frequent_words)

# Write file with all anime urls
# 19139 anime in total, 383 pages with 50 animes

def write_list_anime(path_list_anime, url_of_single_page):
    # Number of total pages
    n_pages = 383 

    with open(path_list_anime, "w", encoding='utf-8') as file:

        for page in tqdm(range(0, n_pages)):

            # URL of the single page 
            url = url_of_single_page + str(page * 50)

            # Get response object
            response = r.get(url)

            # Get html
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all links iterating through the document
            for tag in soup.find_all("tr", class_="ranking-list"):
                a_list = tag.find_all('a', class_="hoverinfo_trigger fl-l ml12 mr8" ,href=True)
                for a in a_list:
                    link = a['href']
                    file.write(str(link) + '\n')

    file.close()
    
# Download all animes html files
def download_html(path_list_anime, path_page_general):
    file = open(path_list_anime, "r", encoding='utf-8')

    # Initialize counters 
    counter_anime = 0
    page = 0

    # Get html file for every anime (url is a line in the file list_anime) 
    for line in tqdm(file):
        counter_anime += 1
        if (counter_anime%50 == 1):
            page +=1
            # Every 50 animes the program stops because of the scraping block of the website
            time.sleep(120)
        response = r.get(line)
        soup = BeautifulSoup(response.text, 'html.parser')

        try:
            os.mkdir(path_page_general + str(page))
        except OSError:
            pass

        with open(path_page_general + str(page) + "/article_" + str(counter_anime) + ".html", "w", encoding='utf-8') as file:
            file.write(str(soup))
        
# Function to retrieve all the information from html
# Retrive information at page i of anime j
def retrive_information(i, j):
        with open(path_page_general + str(i) + "/article_" + str(j) + ".html", "r", encoding='utf-8') as file:
    
            # Read file and parse html code
            content_page = file.read()
            soup = BeautifulSoup(content_page, 'html.parser')
            
            # Retrive information
            
            animeTitle = soup.find_all("h1", class_="title-name h1_bold_none")[0].find("strong").text
        
            animeType = soup.find_all('a', href = re.compile(r'type*'))[0].text.strip()
            
            animeNumEpisode = soup.find(string='Episodes:').next_element.replace("\n","").strip()
            
            animeNumMembers = soup.find(string='Members:').next_element.replace("\n","").replace(',','').replace('#', '').strip()
        
            animeScore = soup.find('span', {"itemprop":"ratingValue"}).text.strip()
            
            animeUsers = soup.find('span', {"itemprop":"ratingCount"}).text.strip()
            
            animeRank = soup.find(string='Ranked:').next_element.replace("\n","").replace('#','').strip()
            
            animePopularity = soup.find(string='Popularity:').next_element.replace("\n","").replace('#', '').strip()
            
            animeDescription = soup.find('p',{'itemprop':'description'}).text.replace('\n', '').replace('  ', '').strip()

    
            # Iterate in di with dates   
            dates_div = soup.find_all("div", class_="spaceit_pad")
            for i in range(0, len(dates_div)):
        
                # Dates
                if dates_div[i].find("span", string='Aired:') != None:
                    dates = dates_div[i].contents[2].strip().split("to")
                    try:
                        releaseDate = datetime.strptime(dates[0].strip(), '%b %d, %Y').date()
                    except:
                        releaseDate = None
                    try:
                        endDate = datetime.strptime(dates[1].strip(), '%b %d, %Y').date()
                    except:  
                        endDate = None
    
            # Anime Description
            animeDescription = soup.find("p", itemprop='description').text
    
            # Related Animes
            try:
                related_anime_table = soup.find("table", class_="anime_detail_related_anime")
                links_list = [link.text for link in related_anime_table.find_all("a")]
                animeRelated = list(set(links_list))
            except:
                animeRelated = None
    
            # Characters
            animeCharacters = soup.find_all("h3", class_="h3_characters_voice_actors")
            animeCharacters = [character.text for character in animeCharacters]
    
            # Voices
            animeVoices = soup.find_all("td", class_="va-t ar pl4 pr4")
            animeVoices = [actor.find("a").text for actor in animeVoices]
    
            a_no = []
            total = []
            result = []
            animeStaff = []
    
            # Staff 
            if animeCharacters == [] and animeVoices == []:
                div = soup.find_all("div", class_="detail-characters-list clearfix")[0]
                tds = div.find_all("td", class_=lambda x: x != 'ac borderClass' and x != 'ac' and x == 'borderClass')
    
                smalls = div.find_all("small")
    
                for td in tds:
                    a_no = set(list(td.find_all("a", class_="fw-n")))
                    total = set(list(td.find_all("a")))
                    if list(total.difference(a_no)) != []:
                        result.append(list(total.difference(a_no)))
    
                for (s, role) in zip(result, smalls):
                    s = str(s).strip("[").strip("]")
                    htmlTag = BeautifulSoup(s, "html.parser")
                    animeStaff.append([htmlTag.text, role.text])
                
            # There are no staff
            elif len(soup.find_all("div", class_="detail-characters-list clearfix")) < 2:
                animeStaff = []
                
            # There are one of animeCharacters or animeVoices
            else:
                div = soup.find_all("div", class_="detail-characters-list clearfix")[1]
                
                tds = div.find_all("td", class_=lambda x: x != 'ac borderClass' and x != 'ac' and x == 'borderClass')

                smalls = div.find_all("small")

                for td in tds:
                    a_no = set(list(td.find_all("a", class_="fw-n")))
                    total = set(list(td.find_all("a")))
                    if list(total.difference(a_no)) != []:
                        result.append(list(total.difference(a_no)))

                for (s, role) in zip(result, smalls):
                    s = str(s).strip("[").strip("]")
                    htmlTag = BeautifulSoup(s, "html.parser")
                    animeStaff.append([htmlTag.text, role.text])
            
        # Create list with overall information
        final_information = [animeTitle, animeType,animeNumEpisode,releaseDate,endDate,
                         animeNumMembers, animeScore,animeUsers,animeRank,animePopularity,
                         animeDescription,animeRelated,animeCharacters,animeVoices,animeStaff]
        return final_information
    
# Create a .tsv file for each anime and a .tsv file with 3 most important information for all: Title, Description, Url 
def create_tsvs(path_list_anime, path_page_general, path_documents_file, path_documents_ex_3):
    # Create .tsv file
    n_pages = 384

    # Collecting all Urls
    with open(path_list_anime, 'r', encoding='utf-8') as f_urls:
        urls = f_urls.readlines()
    f_urls.close

    # Create .tsv with only 3 important fields
    with open(path_documents_file, 'w', encoding='utf-8') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        # Write haeders
        tsv_writer.writerow(['animeTitle','animeDescription','Url'])

    f.close
    
    # Create .tsv unique with only 4 fields related to excercise 3
    with open(path_documents_ex_3, 'w', encoding='utf-8') as f:
        tsv_writer = csv.writer(f, delimiter='\t')

        # Write headers
        tsv_writer.writerow(['animeTitle','animeDescription','Url', 'animeScore'])

    f.close

    # Iterating through pages and then animes
    for i in range(1, n_pages):
        
        # Start and end computed with respect to the 
        start_iterate_anime = (i-1)*50+1
        end_iterate_anime = ((i-1)*50+1)+50
        for j in range(start_iterate_anime, end_iterate_anime):

            # Write for each anime a .tsv file
            with open(path_page_general + str(i) + "/anime_" + str(j)+ ".tsv", 'w', encoding='utf-8') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')

                # Write haeders
                tsv_writer.writerow(['animeTitle', 'animeType','animeNumEpisode','releaseDate','endDate',
                                     'animeNumMembers', 'animeScore','animeUsers','animeRank','animePopularity',
                                     'animeDescription','animeRelated','animeCharacters','animeVoices','animeStaff'])

                # Write information
                information = retrive_information(i, j) 
                tsv_writer.writerow(information)
            out_file.close

            important_information = [information[0],information[10], urls[j]]
            important_information_ex_3 = [information[0],information[10], urls[j], information[6]]

            # Append data to unique .tsv with only 3 important fields
            with open(path_documents_file, 'a', encoding='utf-8') as f:
                tsv_writer = csv.writer(f, delimiter='\t')

                # Write information 
                tsv_writer.writerow(important_information)

            f.close
            
            # Append data to unique .tsv with only 4 fields related to excercise 3
            with open(path_documents_ex_3, 'a', encoding='utf-8') as f:
                tsv_writer = csv.writer(f, delimiter='\t')

                # Write information 
                tsv_writer.writerow(important_information_ex_3)

            f.close

# Create pandas dataframe with important information

def df_from_3_information_file(path_documents_file):
    original_df = pd.read_csv(path_documents_file, sep='\t')
    return original_df

def df_from_4_information_file(path_documents_ex_3):
    original_df = pd.read_csv(path_documents_ex_3, sep='\t')
    return original_df

# Text cleaning functions

def remove_written_mal_rewrite(text):
    return text.replace('[Written by MAL Rewrite]', '')

def remove_contractions(text):
    # Create patterns
    patterns = [
       (r'won\'t', 'will not'),
       (r'can\'t', 'cannot'),
       (r'i\'m', 'i am'),
       (r'(\w+)\'ll', '\g<1> will'),
       (r'(\w+)n\'t', '\g<1> not'),
       (r'(\w+)\'ve', '\g<1> have'),
       (r'(\w+)\'s', '\g<1> is'),
       (r'(\w+)\'re', '\g<1> are'),
    ]
    
    # Compile patterns with re.compile()
    patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    
    # Substitute words
    for (pattern, repl) in patterns:
        text = re.sub(pattern, repl, text)
    return text

def to_lower_case(text):
    return text.lower()

def replace_dashes(text):
    return text.replace('—',' ')

def remove_ordinal_num(text):
    text = re.sub('[\d]+(st|nd|rd|th)', '', text)
    return text

def tokenize_text(text):
    return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

def remove_punctuation(text):
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    new_words = tokenizer.tokenize(text)
    return ' '.join(new_words)

def remove_stopwords(text, stop_words=default_stopwords):
    tokens = [w for w in tokenize_text(text) if w not in stop_words]
    return ' '.join(tokens)
    
def stem_text(text, stemmer=default_stemmer):
    tokens = tokenize_text(text)
    return ' '.join([stemmer.stem(t) for t in tokens])

def clean_text(text):
    
    # Text cleaning
    text = remove_written_mal_rewrite(text)
    text = remove_contractions(text)
    text = to_lower_case(text)
    text = replace_dashes(text)
    text = remove_ordinal_num(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    
    return text

def clean_df(original_df, column_type):
    df_descriptions = original_df[column_type]
    df_descriptions = pd.DataFrame(df_descriptions)
    df_cleaned = df_descriptions.applymap(lambda x : clean_text(x), na_action='ignore')
    return df_cleaned

# Create vocabulary and save it in a file .json

def write_vocabulary(path_vocabulary_file, df_descriptions):    
    # Create a list 
    df_words = df_descriptions.applymap(lambda x : set(tokenize_text(x)), na_action='ignore')
    words_union = set()

    for row in df_words['animeDescription']:
        words_union = words_union.union(set(row))

    words_list = list(words_union)

    # Create vocabulary
    vocabulary = { words_list[i] : i for i in range(len(words_list)) }

    # Write the file
    with open(path_vocabulary_file, 'w', encoding='utf-8') as file:
        json.dump(vocabulary, file)
    file.close
    
# Create index and save it in a file .json

def write_index(vocabulary, path_index_file, df_descriptions):
    # Create the index with (id_word : [doc1, doc2, ...])
    df_words = df_descriptions.applymap(lambda x : set(tokenize_text(x)), na_action='ignore')
    index = {}
    for n_row in range(df_words['animeDescription'].shape[0]):
        for (k,v) in vocabulary.items():
            if v not in index.keys():
                index[v] = []
            if k in df_words['animeDescription'][n_row]:
                # n_row+1 becouse n_anime is n_row_in_dataset + 1
                index[v].append(n_row+1)

    # Write the file
    with open(path_index_file, 'w', encoding='utf-8') as file:
        json.dump(index, file)
    file.close
    
# Functions to read vocabulry and index

def read_vocabulary(path_vocabulary_file):
     # Vocabulary
    with open(path_vocabulary_file, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)
    f.close
    return vocabulary

def read_index(path_index_file):
    # Index
    with open(path_index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)
    f.close
    return index

# Function to execute conjunctive query of excercise 1

def execute_query(query, vocabulary, index, path_documents_file):
    # Query
    query = clean_text(query).split(" ")

    # Trasform every word of the query in the corrispondant id
    id_words_query = []
    for word in query:
        if word in vocabulary.keys():
            id_words_query.append(vocabulary[word])
        else:
            id_words_query = []
            break

    documents_set = set()
    for id_word in id_words_query:
        if documents_set == set():
            documents_set.update(index[str(id_word)])
        else:
            documents_set = documents_set.intersection(set(index[str(id_word)]))
            if documents_set == set():
                break

    documents = list(documents_set)
    documents_in_dataframe = [x-1 for x in documents]

    # Return rows
    original_df = df_from_3_information_file(path_documents_file)
    return original_df.iloc[documents_in_dataframe]

# Inverted index function

def write_inverted_index (original_df, vocabulary, index, column_type, path):
    df = clean_df(original_df, column_type)
    tf = []
    for num_row in range(len(df)):
        doc = df[column_type].loc[num_row].split()
        len_doc = len(doc)
        counter = Counter(doc)
        for item, count in counter.items():
            counter[item] /= len_doc
        tf.append(counter)

    ## per mappare le chiavi di tf in numeri(id)
    tf_new = []
    for d in tf:
        dic = dict((vocabulary[key], value) for (key, value) in d.items())
        tf_new.append(dic)
        
    idf = {}

    numdocs_cont_j = {}
    num_docs = len(df)
    for term_id in index:
        numdocs_cont_j = len(index[term_id])
        idfscore = np.log(num_docs/numdocs_cont_j)
        idf[term_id] = idfscore

    term_id = {}
    for term in idf: #for every term_id
        for i in range(len(tf_new)): #for every doc
            if int(term) in tf_new[i].keys(): #if term_id is the key of the document_i
                tupla = (i+1, idf[term]*tf_new[i][int(term)])

                if term in term_id:
                    term_id[term].append(tupla)
                else:
                    term_id[term] = []
                    term_id[term].append(tupla)
    # Write the file
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(term_id, file)
    file.close
    
# Function that read the inverted index from file .json    
def read_inverted_index(path):
    # Index
    with open(path, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)
    f.close
    return inverted_index

# Function that return the list of indexes of conjunctive query

def docs_contains_query(query, vocabulary, index, path_documents_file):
    subset = execute_query(query, vocabulary, index, path_documents_file)
    indexes = [i for i in subset.index]
    return indexes

# Get the numpy array vector for the query with frequency of words. It has the same lenght of the vocabulary and has number != 0 if the word is present in the query

def get_query_vector(query, vocabulary):
    query = clean_text(query).split(" ")
    query_vector = np.zeros(len(vocabulary.keys()))
    for word in query:
        if word in vocabulary.keys():
            query_vector[int(vocabulary[word])] += 1
    return query_vector

# Same method to get the numpy array vector of the document selected 

def get_doc_vector(n_row_doc, df_cleaned, vocabulary, inverted_index, column_type):
    doc = df_cleaned[column_type].loc[n_row_doc].split(" ")
    doc_vector = np.zeros(len(vocabulary.keys()))
    # iterating in vocabulary of terms
    for term_id, value in inverted_index.items():
        # Iterating in each tuple:
        for pair in value:
            # if the document in which the word is present is the document that i passed to this function 
            if pair[0] == n_row_doc+1:
                # Update doc_vector with the respective tfidf in the position [term_id]  
                doc_vector[int(term_id)] = pair[1]
    return doc_vector

# Function to calculate the cosine similarity between vectors

def calc_score(q_vector, doc_vector):
    # Calculate cosine similarity between query vector and the selected document vector 
    score = np.dot(q_vector, doc_vector)/(math.sqrt(np.linalg.norm(q_vector))*math.sqrt(np.linalg.norm(doc_vector)))
    return score

# Function that returns dataframe with result of similarity query

def execute_similarity_query(query, k, df_cleaned, vocabulary, index, inverted_index, original_df, path_documents_file, column_type):
    # Initialize the heap
    heap = []
    # Query vector
    q_vector = get_query_vector(query, vocabulary)
    
    #Manca la funzionee  docs contain query per confrontare solo con i  documenti di interesse 
    doc_list = docs_contains_query(query,vocabulary, index, path_documents_file)
    # For all the synopsis in the dataframe
    for i in doc_list:
        doc_i_vector = get_doc_vector(i, df_cleaned, vocabulary, inverted_index, column_type)
        score_i = calc_score(q_vector, doc_i_vector)
        
        # Push in the heap the tuple containing (score of i-th document, i)
        if score_i  != 0:
            heapq.heappush(heap, (score_i, i))
    if heap==[]:
        return original_df.iloc[[]]
    else:
        heapq._heapify_max(heap)
        if len(heap) > k:
            heap=[heapq.heappop(heap) for i in range(k)]
        
    # List of indexes of k-documents 
        doc_rows_list = []
        score_list=[]
        for j in range(len(heap)):
            #tuple_element = heapq.heappop(heap)
            doc_rows_list.append(heap[j][1])
            score_list.append(heap[j][0])
        
        sub_df=original_df.iloc[doc_rows_list]
        name_column = 'Similarity ' + column_type
        sub_df[name_column]=np.array(score_list,dtype='float') #purche l ordine con cui printiamo il sub dataset  sia lo  stesso del vettore doc_rows_list
        return  sub_df
    
# Function that write a new vocabulary based on titles words

def write_vocabulary_title(path_vocabulary_title_file, df_title):    
    # Create a list 
    df_words = df_title.applymap(lambda x : set(tokenize_text(x)), na_action='ignore')
    words_union = set()

    for row in df_words['animeTitle']:
        words_union = words_union.union(set(row))

    words_list = list(words_union)

    # Create vocabulary
    vocabulary_title = { words_list[i] : i for i in range(len(words_list)) }

    # Write the file
    with open(path_vocabulary_title_file, 'w', encoding='utf-8') as file:
        json.dump(vocabulary_title, file)
    file.close

# Function that write a new index based on titles vocabulary

def write_index_title(vocabulary_title, path_index_title_file, df_title):
    # Create the index with (id_word : [doc1, doc2, ...])
    df_words = df_title.applymap(lambda x : set(tokenize_text(x)), na_action='ignore')
    index_title = {}
    for n_row in range(df_words['animeTitle'].shape[0]):
        for (k,v) in vocabulary_title.items():
            if v not in index_title.keys():
                index_title[v] = []
            if k in df_words['animeTitle'][n_row]:
                # n_row+1 becouse n_anime is n_row_in_dataset + 1
                index_title[v].append(n_row+1)

    # Write the file
    with open(path_index_title_file, 'w', encoding='utf-8') as file:
        json.dump(index_title, file)
    file.close
    
# Function that execute query with the new similarity score    
    
def execute_new_score_query(query, k, vocabulary, index, invert_index, vocabulary_title, index_title, invert_index_title, 
                            path_documents_file, path_documents_ex_3, prior_title = False, prior_a_score = False):
    
    original_df = df_from_4_information_file(path_documents_ex_3)
    df_cleaned = clean_df(original_df, "animeDescription")
    df_score_desc = execute_similarity_query(query, k, df_cleaned, vocabulary,index , invert_index, original_df, path_documents_file, "animeDescription")
    
    if prior_title and prior_a_score:
        df_cleaned_title = clean_df(original_df, "animeTitle")
        df_score_title = execute_similarity_query(query, k, df_cleaned_title, vocabulary_title,index_title , invert_index_title, original_df, path_documents_file, "animeTitle")
        # Merge the two dataframes to have all scores together
        df_result = df_score_desc.reset_index().merge(df_score_title.reset_index(), how="outer").set_index('index')
        
        # Check if all columns exist after merging
        if 'Similarity animeDescription' not in df_result:
            df_result['Similarity animeDescription'] = 0
        if 'Similarity animeTitle' not in df_result:
            df_result['Similarity animeTitle'] = 0
            
        # Fill NaN values with zeros
        df_result = df_result.fillna(0)
        df_result['FinalScore'] = (np.array(df_result['Similarity animeDescription'])+ 2*np.array(df_result['Similarity animeTitle'])+ (np.array(df_score_desc['animeScore'])/10))/4
        df_result = df_result.sort_values(by=['FinalScore'], ascending=False)
        return df_result.head(k)
    
    elif prior_title:
        df_cleaned_title = clean_df(original_df, "animeTitle")
        df_score_title = execute_similarity_query(query, k, df_cleaned_title, vocabulary_title,index_title , invert_index_title, original_df, path_documents_file, "animeTitle")
        # Merge the two dataframes to have all scores together
        df_result = df_score_desc.reset_index().merge(df_score_title.reset_index(), how="outer").set_index('index')
        
        # Check if all columns exist after merging
        if 'Similarity animeDescription' not in df_result:
            df_result['Similarity animeDescription'] = 0
        if 'Similarity animeTitle' not in df_result:
            df_result['Similarity animeTitle'] = 0
            
        # Fill NaN values with zeros
        df_result = df_result.fillna(0)
        df_result['FinalScore'] = (np.array(df_result['Similarity animeDescription'])+ 2*np.array(df_result['Similarity animeTitle']))/3
        df_result = df_result.sort_values(by=['FinalScore'], ascending=False)
        return df_result.head(k)
    
    elif prior_a_score:
        # Check if all columns exist
        if 'Similarity animeDescription' not in df_result:
            df_result['Similarity animeDescription'] = 0
            
        df_score_desc['FinalScore'] = (np.array(df_score_desc['Similarity animeDescription'])+ (np.array(df_score_desc['animeScore'])/10))/2
        df_score_desc = df_score_desc.sort_values(by=['FinalScore'], ascending=False)
        return df_score_desc.head(k)
    
    else:
        df_score_desc = df_score_desc.sort_values(by=['Similarity animeDescription'], ascending=False)
        return df_score_desc.head(k)