import numpy as np
import pandas as pd
import spacy_stanza
import spacy
import stanza
import medspacy
import pickle
import datefinder
import re
import random
import string
from pathlib import Path
from medspacy.context import ConTextRule
from medspacy.ner import TargetRule
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from negspacy.negation import Negex
from genderComputer import GenderComputer


# Model Path
d_path = Path.cwd() / 'src' / 'data'

"""
Load basic model components
    1. en_core_web_sm (spacy)
    2. en_core_sci_sm (scispaCy)
    3. en_ner_bc5cdr_md (scispaCy)
    4. en_core_med7_lg (med7)
    5. medspaCy (ConText Rules)
    6. stanza (StanfordNLP)
    7  obi/deid_bert_i2b2 (transformers model)
    7. obi/deid_bert_i2b2 (transformers tokenizer)
    8. Medical Dicts
"""
# Basic spaCy
en_nlp = spacy.blank('en')
en_nlp.add_pipe('sentencizer')

# De-Identication
tokenizer = AutoTokenizer.from_pretrained("obi/deid_bert_i2b2")
model = AutoModelForTokenClassification.from_pretrained("obi/deid_bert_i2b2")
deid_nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="max")
male_words = ['man', 'manlike', 'male', 'gentleman', 'boy', 'manful', 'masculine', 'dude', 'guy']
male_initial = ['sir', 'mr.', 'mister']
female_words = ['woman', 'feminine', 'female', 'girl', 'gentlewoman']
female_initial = ['miss', 'mrs.', 'madam', 'madame']

# Problems
stanza.download('en', package='mimic', processors={'ner': 'i2b2'}, verbose=False)
spacy_stanza_nlp = spacy_stanza.load_pipeline('en', package='mimic', processors={'ner': 'i2b2'}, use_gpu=False, verbose=False)
dis_nlp = spacy.load("en_ner_bc5cdr_md")
meds_nlp = medspacy.load()

# Attetion
scispacy_nlp = spacy.load("en_core_sci_sm")
scispacy_nlp.add_pipe("negex")

# Allergies
med7_nlp = spacy.load("en_core_med7_lg")
with open(d_path / 'allergens_list.pkl', 'rb') as f:
    allergens_l = pickle.load(f)
    
all_nlp = medspacy.load()
all_terms = ['allergies', 'allergy', 'allergic', 'hypersensitivity', 'hypersensitive', 'sensitive', 'sensitivity']
rules = [TargetRule(allergen, 'ALLERGEN') for allergen in allergens_l]
all_nlp.get_pipe('medspacy_target_matcher').add(rules)
context_rules = [
    ConTextRule("<ALLERGY_TERM>", "ALLERGY", 
                rule="FORWARD",
            pattern=[
                {"LOWER": {"IN": all_terms}},
            ])
]
all_nlp.get_pipe('medspacy_context').add(context_rules)
    
def join_ents(text,ents):
    new_ents = []
    for idx, ent in enumerate(ents):
        if not idx:
            new_ents.append(ent)
            continue
        
        ant_ent = new_ents[-1]
        interval = text[ant_ent['end']:ent['start']]
        if len(interval)<=1:
            if ant_ent['entity_group']==ent['entity_group']:
                new_start = ant_ent['start']
                new_end = ent['end']
                ent_text = text[new_start:new_end]
                new_ents[-1]['start'] = new_start
                new_ents[-1]['end'] = new_end
                new_ents[-1]['word'] = ent_text
                continue
        
        new_ents.append(ent)
    
    return new_ents

def get_deid(text):
    ents = deid_nlp(text)
    ents = join_ents(text, ents)
    deid_d = {}
    deid_ents = ['DATE', 'PATIENT', 'HOSP', 'STAFF', 'AGE', 'LOC']
    gender = ''
    gc = GenderComputer()
    for i in ents:
        ent_g = i['entity_group']
        value = i['word']
        if ent_g not in deid_ents:continue
        if ent_g=='AGE':
            # Gender
            hypo_gender=re.sub("[^FfMm]", "", value)
            if hypo_gender:
                gender = (hypo_gender, 'F' if hypo_gender in ['f','F'] else 'M')
            
            # Age
            value=int(re.sub("[^0-9]", "", value))
            
        deid_d[ent_g] = value if not deid_d.get(ent_g) else deid_d[ent_g]
    
    
    # Gender
    if not gender:
        if 'PATIENT' in deid_d.keys():
            pat_name = deid_d['PATIENT']
            res = gc.resolveGender(pat_name,None)
            if res:
                res = 'F' if res=='female' else 'M'
                gender = (pat_name, res)
            if not gender:
                patient_start = text.index(pat_name)
                target = text[:patient_start].lower()
                for i in male_initial:
                    if i in target:
                        gender = (i, 'M')
                for i in female_initial:
                    if i in target:
                        gender = (i, 'F')
        else:
            target = text.lower().split()
            for i in male_words:
                if i in target:
                    gender = (i, 'M')
            for i in female_words:
                if i in target:
                    gender = (i, 'F')
    if gender:
        deid_d['GENDER']=gender
    return deid_d

def get_context(text, problems):
    pres_problems = []
    hist_problems = []
    fam_problems = []
    neg_problems = []
    rules = [TargetRule(problem[0], 'CONDITION') for problem in problems]
    meds_nlp.get_pipe('medspacy_target_matcher').add(rules)
    doc = meds_nlp(text)
    for ent in doc.ents:
        if ent._.is_negated:
            neg_problems.append((ent.text, ent.start_char))
        elif ent._.is_historical:
            hist_problems.append((ent.text, ent.start_char))
        elif ent._.is_family:
            fam_problems.append((ent.text, ent.start_char))
        else:
            pres_problems.append((ent.text, ent.start_char))
    
    return pres_problems, hist_problems, fam_problems, neg_problems


def get_problems(sentence):
    doc_stanza = spacy_stanza_nlp(sentence)
    doc_dis_spacy = dis_nlp(sentence)
    problems = [(ent.text, ent.start_char) for ent in doc_stanza.ents if ent.label_=="PROBLEM"]
    diseases = [(ent.text, ent.start_char) for ent in doc_dis_spacy.ents if ent.label_=="DISEASE"]
    
    for dis, idx in diseases:
        if any((dis.lower() in problem[0].lower()) for problem in problems):
            continue
        problems.append((dis, idx))
        
    return get_context(sentence, problems)

def get_attention_words(sentence):
    doc = scispacy_nlp(sentence)
    attentions = []
    negs = []
    for ent in doc.ents:
        if not ent._.negex:
            attentions.append((ent.text, ent.start_char))
        else:
            negs.append((ent.text, ent.start_char))
            
    return attentions, negs

def get_allergens(sentence):
    allergens = []
    negs = []
    doc_en = en_nlp(sentence)
    doc_med = med7_nlp(sentence)
    chemicals = [ent.text for ent in doc_med.ents if ent.label_=="DRUG"]
    rules = [TargetRule(chemical, 'ALLERGEN') for chemical in chemicals]
    all_nlp.get_pipe('medspacy_target_matcher').add(rules)
    
    for sent in doc_en.sents:
        doc_all = all_nlp(sent.text)
        for ent in doc_all.ents:
            if ent._.is_negated:
                negs.append((ent.text, ent.start_char))
            else:
                if ent._.modifiers:
                    if ent._.modifiers[0].category=="ALLERGY":
                        allergens.append((ent.text, ent.start_char))
    return allergens, negs

def get_ents_input_text(text):
    w_detected = []
    neg_problems = []
    
    ## De-identification
    input_d = get_deid(text)
    w_detected+=input_d.values()
    
    ## Allergens
    allergens, negs = get_allergens(text)
    for allergen, idx in allergens:
        w_detected+=[allergen]
    input_d['ALLERGEN'] = allergens + negs
    all_negs = negs
    
    ## Problems
    problems, hist_problems, fam_problems, negs = get_problems(text)
    new_problems = []
    new_hist_problems = []
    new_fam_problems = []
    cur_detected=" ".join(list(map(lambda x: str(x).lower(),w_detected))).split()
    for problem, idx in problems:
        words_l = problem.lower().split()
        if any((i in cur_detected) for i in words_l):
            continue
        else:
            new_problems.append((problem, idx))
    for hist_problem, idx in hist_problems:
        words_l = hist_problem.lower().split()
        if any((i in cur_detected) for i in words_l):
            continue
        else:
            new_hist_problems.append((hist_problem, idx))
    
    for fam_problem, idx in fam_problems:
        words_l = fam_problem.lower().split()
        if any((i in cur_detected) for i in words_l):
            continue
        else:
            new_fam_problems.append((fam_problem, idx))
    
    for problem, idx in new_problems:
        w_detected+=[problem]
    for hist_problem, idx in new_hist_problems:
        w_detected+=[hist_problem]
    for fam_problem, idx in new_fam_problems:
        w_detected+=[fam_problem]
    input_d['PROBLEM'], input_d['HIST_PROBLEM'], input_d['FAM_PROBLEM'] = new_problems, new_hist_problems, new_fam_problems

    # Remove negated from allergens
    for neg, idx in negs:
        if not any((neg in _neg[0]) for _neg in all_negs):
            neg_problems.append((neg, idx))
    
    ## Attention
    cur_detected=" ".join(list(map(lambda x: str(x).lower(),w_detected))).split()
    attentions, negs = get_attention_words(text)
    new_attentions = []
    for attention, idx in attentions:
        words_l = attention.lower().split()
        if any((i in cur_detected) for i in words_l):
            continue
        else:
            new_attentions.append((attention, idx))
    input_d['ATTENTION'] = list(set(new_attentions))
    
    # Remove negated from allergens and problems
    for neg, idx in negs:
        if not any((neg in _neg[0]) for _neg in neg_problems):
            neg_problems.append((neg, idx))
    input_d['NEGATED'] = neg_problems
    
    # Remove negated words from problems
    final_problems = []
    for problem, idx in new_problems:
        if not any((_neg[0] in problem) for _neg in neg_problems):
            final_problems.append((problem, idx))

    input_d['PROBLEM'] = final_problems
    
    # Date
    matches = datefinder.find_dates(text, source=True, index=True, strict=True)
    if matches:
        for match in matches:
            date, source, start = match
            start = start[0]+1
            source = source.split()[0]
            input_d['DATE'] = (date, source, start)
            break
    return input_d

def get_entity_options(ents_l):
    colors={}
    for ent in ents_l:
        colors[ent]=("#"+''.join([random.choice(string.hexdigits) for i in range(6)])).upper()
    
    options = {"ents": ents_l, "colors": colors, "distance": 500}
    return options


def get_ents_input_text_vis(res, text, ent_style='span'):
    _str_type = ['PATIENT', 'HOSP', 'STAFF', 'AGE', 'LOC']
    _str_or_tuple_type = ['DATE']
    _tuple_type = ['GENDER']
    _list_type = ['PROBLEM', 'HIST_PROBLEM', 'FAM_PROBLEM','NEGATED', 'ALLERGEN', 'ATTENTION']
    doc = en_nlp(text)
    spans_list = []
    ents_l = []
    
    for ent_name, value in res.items():
        if ent_name in _str_type:
            value = str(value)
            start = text.index(value)
            end = start + len(value)
            tok_start = len(en_nlp(text[:start]))
            tok_end = tok_start + len(en_nlp(text[start:end]))
            span = doc[tok_start:tok_end]
            span.label_ = ent_name
            ents_l.append(ent_name)
            spans_list.append(span)
        if ent_name in _str_or_tuple_type:
            if isinstance(value, str):
                start = text.index(value)
            if isinstance(value, tuple):
                start = value[2]
                value = value[1]
            end = start + len(value)
            tok_start = len(en_nlp(text[:start]))
            tok_end = tok_start + len(en_nlp(text[start:end]))
            span = doc[tok_start:tok_end]
            span.label_ = ent_name
            ents_l.append(ent_name)
            spans_list.append(span)
        if ent_name in _tuple_type:
            ent_name_t = f"{'FEM' if value[1]=='F' else 'MALE'}"
            start = text.index(value[0])
            end = start + len(value)
            tok_start = len(en_nlp(text[:start]))
            tok_end = tok_start + len(en_nlp(text[start:end]))
            span = doc[tok_start:tok_end]
            span.label_ = ent_name_t
            ents_l.append(ent_name_t)
            spans_list.append(span)
        if ent_name in _list_type:
            for i, idx in value:
                start = idx if ent_name!='ALLERGEN' else text.index(i)
                end = start + len(i)
                tok_start = len(en_nlp(text[:start]))
                tok_end = tok_start + len(en_nlp(text[start:end]))
                span = doc[tok_start:tok_end]
                span.label_ = ent_name
                ents_l.append(ent_name)
                spans_list.append(span)
    
    options = get_entity_options(ents_l)
    if ent_style=='span':
        doc.spans["sc"] = spans_list
    else:
        doc.ents = spans_list
    
    return doc, options