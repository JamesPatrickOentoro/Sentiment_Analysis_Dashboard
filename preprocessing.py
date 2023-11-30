import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import torch
import emoji
from googletrans import Translator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nltk
nltk.download('stopwords')

def handle_missing_value(df): #Success
    # Erase baris dengan label atau komen kosong
    # df_raw = df_raw[df_raw['Label (1,0,-1)'].notna()]
    df = df[df['comment'].notna()]
    df = df[df['comment']!='']
    df = df.reset_index(drop=True)
    return df

def format_bank_name(df): #Success
    bank_acc = {
        "Instagram" : {
            "goodlifebca" : "BCA",
            "bankmandiri" : "Mandiri",
            "bankbri_id" : "BRI",
            "bni46" : "BNI",
            "bankmegaid" : "MEGA",
            "ocbc_nisp" : "OCBC",
            "cimb_niaga" : "CIMB"
        },
        "Facebook" : {
            "BankBCA" : "BCA",
            "bankmandiri" : "Mandiri",
            "BRIofficialpage" : "BRI",
            "BNI" : "BNI",
            "BankMegaID" : "MEGA",
            "bankocbcnisp" : "OCBC",
            "CIMBIndonesia" : "CIMB"
        },
        "Tiktok" : {},
        "Twitter" : {
            "BankBCA" : "BCA",
            "bankmandiri" : "Mandiri",
            "BANKBRI_ID" : "BRI",
            "BNI" : "BNI",
            "BankMegaID" : "MEGA",
            "bankocbcnisp" : "OCBC",
            "CIMBNiaga" : "CIMB"
        },
        "Youtube" : {}
    }

    for index, row in df.iterrows():
        original_name = row['bank']
        platform = row['platform']
        df.at[index, 'bank'] = bank_acc[platform][original_name]
    print(df)
    return df

def format_data_type_string(df): #Success
    for column in df.columns:
        df[column] = df[column].astype(str)
    return df

def remove_at(df): #Success
    df['comment'] = df['comment'].str.replace(r'@\w+\b', '', regex=True)
    return df

def remove_hashtag(df): #Success
    df['comment'] = df['comment'].str.replace(r'#\w+\b', '')
    return df

def remove_http(df): #Success
    df['comment'] = df['comment'].str.replace(r'https\S*', '')
    return df

def remove_duplicate_comment(df): #Success
    df['comment'] = df['comment'].drop_duplicates()
    return df

def change_slang_words(df, slang_words_dictionary):
    list_sentence_train = []
    for sentence in tqdm(df['comment']) :
        cleaned_sentence = [slang_words_dictionary[word] if word in list(slang_words_dictionary.keys()) else word for word in str(sentence).split()]
        list_sentence_train.append(' '.join(cleaned_sentence))
    df['comment'] = list_sentence_train
    return df

def add_brackets_to_emoji(text):
    emojified_text = emoji.emojize(text)
    modified_text = ""
    for char in emojified_text:
        if emoji.is_emoji(char):
            modified_text += " [" + char + "] "
        else:
            modified_text += char
    return modified_text

def translate_emoji(text):
    text = add_brackets_to_emoji(text)
    # print(text)
    # demojize first
    text = emoji.demojize(text,delimiters=("", ""))

    # lower text
    cleaned = re.sub('[^a-zA-Z0-9\[\]]+',' ',text).lower()

    # remove 2 letter words
    shortword = re.compile(r'\W*\b\w{1,2}\b')
    cleaned = re.sub(shortword, '', cleaned)

    # double whitespace to single
    cleaned = re.sub('[ ]+',' ',cleaned)

    return cleaned

def trans_to_id(text, translator):
    try:
        res = translator.translate(text,src='en',dest='id').text
        return res
    except:
        return ''

def find_en(df, translator):
    result = []
    datas = df['comment'].reset_index(drop=True)
    # print(datas)
    # print(len(datas))
    for i in tqdm(range(len(datas))):
        curr_word = ''
        # print(i)
        # print(datas[i])
        for word in re.split(r'\s+(?![^[\]]*\])', datas[i]):
            word = word.strip('[]')
            # print(word)
            if word != '' and word != None:
                if translator.detect(word) != None and translator.detect(word).lang == 'en':
                    curr_word += ' '
                    curr_word += trans_to_id(word, translator)
                else:
                    curr_word += ' '
                    curr_word += word
        result.append(curr_word)
        # print(curr_word)
    df['comment'] = result
    return df

if __name__=="__main__":
    slang_words2 = pd.read_csv('kamus_slang.csv', header=0)
    slang_words2 = slang_words2[['informal', 'formal']]
    
    # Slang words
    slang_words = dict(slang_words2.values)

    # Stopwords
    additional_stop = ['nya','yg','ga','gk','tp','nih','noh','lah','dong','pa','yuk','gak','ya','sih','yaa','aja', 'min', 'bca','brimo','biar','kak','blu','mega','allo','bank','bca','btn']
    all_stopwords = stopwords.words('indonesian') + additional_stop

    # Translator
    translator = Translator()

    df = pd.read_excel('scrap_ig_2023-11-20.xlsx')
    print(df['comment'])
    # print(remove_at(df)['comment'])
    # print(remove_hashtag(df)['comment'])
    # print(remove_http(df)['comment'])
    # print(remove_duplicate_comment(df)['comment'])
    # print(handle_missing_value(df)['comment'])
    # print(change_slang_words(df)['comment'])
    # print(change_slang_words(df,slang_words)['comment'])
    # print(find_en(df,translator)['comment'])
    # print(df['comment'].map(translate_emoji))
