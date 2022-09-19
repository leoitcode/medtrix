import pandas as pd
import re
import random
import numpy as np
from numpy.linalg import norm
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from transformers import AutoTokenizer, AutoModel
from ast import literal_eval


# Model Path
d_path = Path.cwd() / 'src' / 'data'

"""
Load basic model components
    1. df_struct (MIMIC-III Database)
    2. df_struct_lemma (MIMIC-III Database Lemmatization)
    3. df_struct_fam (MIMIC-III Database Family History)
    4. GanjinZero/UMLSBert_ENG (transformers model)
    5. GanjinZero/UMLSBert_ENG (transformers tokenizer)
"""
df_struct = pd.read_csv(d_path / "df_struct.csv")
df_struct_lemma = pd.read_csv(d_path / "df_struct_lemma.csv")
df_struct_text = pd.read_csv(d_path / "df_struct_text.csv")
df_struct_fam = pd.read_csv(d_path / "df_struct_fam.csv")
coder_model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
coder_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')

# Pattern to get MIMIC Topics
add_topics = ['facility', 'HISTORY  OF  THE  PRESENT  ILLNESS(?=\:)', 'Admission Date(?=\:)', 'Discharge Date(?=\:)', 'Sex(?=\:)', 'Chief Complaint(?=\:)', 'Addendum(?=\:)', '(?i)HISTORY OF PRESENT ILLNESS(?=\:)']
pattern = re.compile(f"((?<=\\n\\n)[\w\s]+(?=\:))|{'|'.join(add_topics)}", flags=0)
hpi_p = re.compile("\[\*\*[^\[]*\*\*\]", flags=0)
lemmatizer = WordNetLemmatizer()

def get_topics_text(text):
    topics = []
    positions = []
    sections_text = {}
    for m in pattern.finditer(text):
        s = m.group().replace('\n','')
        s = "_".join(s.lower().split())
        topics.append(s)
        positions.append((m.span()[0], m.span()[1]+2))
    for i, topic in enumerate(topics):
        start = positions[i][1]
        try:
            end = positions[i+1][0]
        except:
            end = len(text)-1
        sections_text[topic]=text[start:end].replace('\n',' ')
        
    return sections_text

def UMLSBert_similarity(sent1, sent2):
    inputs_1 = coder_tokenizer(sent1, return_tensors='pt')
    inputs_2 = coder_tokenizer(sent2, return_tensors='pt')

    sent_1_embed = np.mean(coder_model(**inputs_1).last_hidden_state[0].detach().numpy(), axis=0)
    sent_2_embed = np.mean(coder_model(**inputs_2).last_hidden_state[0].detach().numpy(), axis=0)
    
    return np.dot(sent_1_embed, sent_2_embed)/(norm(sent_1_embed)* norm(sent_2_embed))

def get_jaccard_sim(words_l_1, words_l_2, prop=0.5):
    words_l_1 = set(words_l_1)
    words_l_2 = set(words_l_2)
    len_1 = len(words_l_1)
    len_2 = len(words_l_2)
    f = (len_1 / len_2) * prop
    a = words_l_1
    b = words_l_2
    c = a.intersection(b)
    res = float(len(c)) / (len(a) + len(b) - len(c))
    res_f = res + (res * f)
    return res_f

def lemmatizer_l(sentence_l):
    new_sentence_l = []
    for sentence in sentence_l:
        word_l = word_tokenize(sentence)
        new_word_l = [lemmatizer.lemmatize(word).lower() for word in word_l]
        new_sentence_l.append(" ".join(new_word_l))
        
    return new_sentence_l

def coeff(exp1, exp2, neg=False):
    exp1 = literal_eval(exp1)
    if (not exp1) or (not exp2):
        return 0
    jacc = get_jaccard_sim(exp1, exp2)
    return jacc if not neg else -jacc

def umls_coeff(exp1, exp2, neg=False):
    exp1 = literal_eval(exp1)
    exp1_s = " ".join(exp1)
    exp2_s = " ".join(exp2)
    try:
        umls = UMLSBert_similarity(exp1_s, exp2_s)
    except:
        exp1_s = " ".join(exp1[:450])
        exp2_s = " ".join(exp2[:450])
        umls = UMLSBert_similarity(exp1_s, exp2_s)
        
    return umls if not neg else -umls

def get_similar_document(ents_d):
    prob_cols = ['chief_complaint', 'history_of_present_illness', 'brief_hospital_course', 'hospital_course', 'discharge_diagnosis']
    att_cols = ['social_history']
    hist_cols = ['past_medical_history']
    
    ## PROBLEM SCORE, ATT SCORE, HIST SCORE, NEG
    cols_d = {}
    cols_d['PROBLEM'] = prob_cols
    cols_d['ATTENTION'] = att_cols
    cols_d['HIST_PROBLEM'] = hist_cols
    cols_d['NEGATED'] = prob_cols + att_cols + hist_cols
    
    # Gender
    gender = ents_d.get('GENDER')
    if gender:
        gender = gender[1]
        df_struct_target = df_struct_lemma[df_struct_lemma.sex==gender]
    else:
        df_struct_target = df_struct_lemma
    
    # Age
    age = ents_d.get('AGE')
    if age:
        df_struct_target = df_struct_target[df_struct_target.age.between(age-5, age+5)]
    else:
        pass
    
    # Problems, Historical Problems, Attention, Negated
    idx_subj = {}
    for subject, cols in cols_d.items():
        exp2 = lemmatizer_l([ent for ent, idx in ents_d[subject]])
        if subject!='NEGATED':
            for col in cols:
                df_struct_target[f'coeff_{subject}_{col}'] = df_struct_target[col].apply(coeff, exp2=exp2, neg=False)
        else:
            for col in cols:
                df_struct_target[f'coeff_{subject}_neg_{col}'] = df_struct_target[col].apply(coeff, exp2=exp2, neg=True)
    for subject in cols_d.keys():
        if subject=='NEGATED':continue
        n_rows = 5
        target_cols = df_struct_target.columns[df_struct_target.columns.str.startswith(f"coeff_{subject}")].to_list()
        target_cols += df_struct_target.columns[df_struct_target.columns.str.startswith(f"coeff_{subject}_neg")].to_list()
        df_struct_target[f'coeff_total_{subject}'] = df_struct_target[target_cols].sum(axis=1)
        idx_subj[subject] = df_struct_target.sort_values(by=f'coeff_total_{subject}', ascending=False).head(n_rows).index
    
    idx_subj_f = {}
    for subject, idxs in idx_subj.items():
        if not ents_d[subject]:continue
        df_struct_target = df_struct.loc[idxs]
        exp2 = [ent for ent, idx in ents_d[subject]]
        target_cols = cols_d[subject]
        for col in target_cols:
            df_struct_target[f'coeff_{subject}_{col}'] = df_struct_target[col].apply(umls_coeff, exp2=exp2, neg=False)
        target_cols = df_struct_target.columns[df_struct_target.columns.str.startswith(f"coeff_{subject}")].to_list()
        df_struct_target[f'coeff_total_{subject}'] = df_struct_target[target_cols].sum(axis=1)
        idxs_f = df_struct_target.sort_values(by=f'coeff_total_{subject}', ascending=False).head(2).index
        idx_f = random.choice(idxs_f)
        idx_subj_f[subject] = idx_f
    
    # Select the main text
    idx_doc = idx_subj_f.get('PROBLEM')
    if idx_doc:
        idx_doc_problem = idx_doc
        text_selected = df_struct_text.loc[idx_doc]['text']
    else:
        idx_doc = idx_subj_f.get('ATTENTION')
        if idx_doc:
            text_selected = df_struct_text.loc[idx_doc]['text']
        else:
            idx_doc = idx_subj_f.get('HIST_PROBLEM')
            if idx_doc:
                text_selected = df_struct_text.loc[idx_doc]['text']
            else:
                text_selected = df_struct_text.sample(n=1)['text'].to_list()[0]
                                                                        
    # Select Social History Text
    idx_doc_ = idx_subj_f.get('ATTENTION')
    if idx_doc:
        text_selected_att = df_struct_text.loc[idx_doc]['text']
    else:
        text_selected_att = text_selected

    # Select Past Medical History
    idx_doc = idx_subj_f.get('HIST_PROBLEM')
    if idx_doc:
        text_selected_hist = df_struct_text.loc[idx_doc]['text']
    else:
        text_selected_hist = text_selected
    
    sections_text = get_topics_text(text_selected)
    sections_text_att = get_topics_text(text_selected_att)
    sections_text_hist = get_topics_text(text_selected_hist)
    
    sections_text['social_history'] = sections_text_att.get('social_history')
    sections_text['past_medical_history'] = sections_text_hist.get('past_medical_history')
    
    # Family History
    subject = 'FAM_PROBLEM'
    col = 'family_history'
    if subject in ents_d.keys():
        exp2 = [ent for ent, idx in ents_d[subject]]
        df_struct_fam[f'coeff_{subject}_{col}'] = df_struct_fam[col].apply(coeff, exp2=exp2, neg=False)
        idxs_f = df_struct_fam.sort_values(by=f'coeff_{subject}_{col}', ascending=False).head(2).index
        idx_f = random.choice(idxs_f)
        sections_text['family_history'] = df_struct_fam.loc[idx_f]['text']
    
    # Replace Allergies and Chief Complaint
    allergies = ents_d.get('ALLERGEN')
    if allergies:
        allergies = [ent for ent, idx in allergies]
    problems = ents_d.get('PROBLEM')
    if problems:
        problems = [ent for ent, idx in problems]
    sections_text['allergies'] = ", ".join(allergies) if allergies else sections_text.get('allergies', '')
    sections_text['chief_complaint'] = ", ".join(problems) if problems else sections_text.get('chief_complaint', '')
    
    return sections_text