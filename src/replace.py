import re
import datefinder
from scripts.placeholdermapper import PlaceholderMapper

plh = PlaceholderMapper()

# Pattern to get Topics
add_topics = ['facility', 'HISTORY  OF  THE  PRESENT  ILLNESS(?=\:)', 'Admission Date(?=\:)', 'Discharge Date(?=\:)', 'Sex(?=\:)', 'Chief Complaint(?=\:)', 'Addendum(?=\:)', '(?i)HISTORY OF PRESENT ILLNESS(?=\:)']
pattern = re.compile(f"((?<=\\n\\n)[\w\s]+(?=\:))|{'|'.join(add_topics)}", flags=0)

# Patterns
hpi_p = re.compile("\\[\*\*([^\[]*)\*\*\]", flags=0) # Any PHI Label
firstn_p = re.compile("\[\*\*Known firstname \d+\*\*\]", flags=0) # Patient's First Name
lastn_p = re.compile("\[\*\*Known lastname \d+\*\*\]", flags=0) # Patient's Last Name
hosp_p = re.compile("\[\*\*Hospital1 18\*\*\]") # Hospital Name
date_p = re.compile("\[\*\*(\d+)-(\d+)-(\d+)\*\*\]") # General Date

# Pattern Age
year_old_l_1 = ['yo', 'y/o', 'year old', 'year-old', 'year-old', 'y.o', 'year o', 'y old']
year_old_l_2 = ['F','M']
year_old_p_1 = re.compile(f"(\d+)(?=\s*\-*({'|'.join(year_old_l_1)}))")
year_old_p_2 = re.compile(f"(\d+)(?=\s*({'|'.join(year_old_l_2)}))")


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

def replace_age(sentence, age):
    if sentence:
        target = " ".join(sentence.split()[:50])
        res = re.search(year_old_p_1, target)
        if res:
            sentence = sentence.replace(res.group(), age)
            return sentence
        else:
            res = re.search(year_old_p_2, target)
            if res:
                sentence = sentence.replace(res.group(), age)
                return sentence
            else:
                if ('999' in sentence) and (age>=90):
                    sentence = sentence.replace('999', age)
        
    return sentence

def fake_phi_labels(sections_text, **kwargs):
    age = kwargs.get('AGE')
    dr_name = kwargs.get('STAFF')
    patient_name = kwargs.get('PATIENT')
    hosp_name = kwargs.get('HOSP')
    adm_date = kwargs.get('DATE')
    hc_topic = ''
    for topic in sections_text.keys():
        if 'hospital_course' in topic:
            hc_topic = topic
            break
    
    ## Age
    if age:
        age = str(age)
        hpi = sections_text.get('history_of_present_illness', '')
        hc = sections_text.get(hc_topic, '')
        if hpi:
            res = replace_age(hpi, age)
            sections_text['history_of_present_illness'] = res
        if hc:
            res = replace_age(hc, age)
            sections_text[hc_topic] = res

    
    ## Doctor Name
    if dr_name:
        att_text = sections_text.get('attending')
        if att_text:
            for hpi in hpi_p.finditer(att_text):
                att_text = att_text.replace(str(hpi.group()), dr_name)
                break
            sections_text['attending'] = att_text
    
    ## Patient Name
    if patient_name:
        for section in sections_text.keys():
            if not sections_text.get(section):continue
            pat_text = sections_text[section]
            for lastn in lastn_p.finditer(pat_text):
                pat_text = pat_text.replace(str(lastn.group()), patient_name)
            for firstn in firstn_p.finditer(pat_text):
                pat_text = pat_text.replace(str(firstn.group()), patient_name)
            sections_text[section]=pat_text
    
    ## Hospital Name
    if hosp_name:
        fac_text = sections_text.get('facility')
        if fac_text:
            for hpi in hpi_p.finditer(fac_text):
                if 'hospital' in hpi.group().lower():
                    fac_text = fac_text.replace(str(hpi.group()), hosp_name)
            sections_text['facility'] = fac_text
        
        for section, text in sections_text.items():
            for hosp in hosp_p.finditer(text):
                text = text.replace(str(hosp.group()), hosp_name)
            sections_text[section]=text
            
    ## Admission Date and Other Dates
    if adm_date:
        if isinstance(adm_date, str):
            matches = datefinder.find_dates(text)
            if matches:
                for date in matches:
                    day = date.day
                    month = date.month
                    year = date.year
                    break
        
        if isinstance(adm_date, tuple):
            date = adm_date[0]
            day = date.day
            month = date.month
            year = date.year
            
        new_adm_date = f"{year}-{month}-{day}"    
        adm_text = sections_text.get('admission_date')
        if adm_text:
            for hpi in date_p.finditer(adm_text):
                adm_text = adm_text.replace(str(hpi.group()), new_adm_date)
                adm_year_fake = int(hpi.group(1))
                adm_month_fake = int(hpi.group(2))
                adm_day_fake = int(hpi.group(3))
                break

            sections_text['admission_date'] = adm_text

            for section, text in sections_text.items():
                replaces = []
                for hpi in date_p.finditer(text):
                    date_y = int(hpi.group(1))
                    date_m = int(hpi.group(2))
                    date_d = int(hpi.group(3))
                    if year:
                        diff = date_y - adm_year_fake
                        new_y = year + diff
                    else:
                        new_y = 0
                    if month:
                        diff = date_m - adm_month_fake
                        new_m = month + diff
                        if new_m < 0:
                            new_y-=1
                            new_m = 12 - abs(new_m)
                    else:
                        new_m = 0
                    if day:
                        diff = date_d - adm_day_fake
                        new_d = day + diff
                        if new_d < 0:
                            new_m-=1
                            new_d = 30 - abs(new_d)
                    else:
                        new_d = 0
                    new_date = f"{new_y}-{new_m}-{new_d}"
                    replaces.append((hpi.group(0), new_date))
                for orig, replace in replaces:
                    text = text.replace(orig, replace)
                sections_text[section] = text
    
    ## Other PHI Labels
    for section, text in sections_text.items():
        replaces = []
        for hpi in hpi_p.finditer(text):
            new_text = plh.get_mapping(hpi.group())
            replaces.append((hpi.group(), new_text))
        for orig, replace in replaces:
            text = text.replace(orig, replace)
        sections_text[section] = text

    ## BirthDay
    adm_text = sections_text.get('admission_date')
    adm_year = ''
    if adm_text:
        for hpi in date_p.finditer(adm_text):
            adm_year = int(hpi.group(1))
            break
    
    if not adm_year:
        adm_year = int(sections_text.get('admission_date').split("-")[0].strip())
    
    if age:
        y_birthday = adm_year-int(age)
    else:
        hpi = sections_text.get('history_of_present_illness', '')
        hc = sections_text.get(hc_topic, '')
        for sentence in [hpi, hc]:
            target = " ".join(sentence.split()[:50])
            res = re.search(year_old_p_1, target)
            if res:
                age = res.group()
                y_birthday = adm_year-int(age)
                break
            else:
                res = re.search(year_old_p_2, target)
                if res:
                    age = res.group()
                    y_birthday = adm_year-int(age)
                    break
                else:
                    if ('999' in sentence):
                        y_birthday = adm_year-90
                        break
                    else:
                        y_birthday = None
    
    if y_birthday:
        birthday_text = sections_text.get('date_of_birth')
        if birthday_text:
            for hpi in date_p.finditer(birthday_text):
                birth_year = int(hpi.group(1))
                birthday_text = birthday_text.replace(birth_year, y_birthday)
                sections_text['date_of_birth'] = birthday_text
        
    return sections_text

def pretty_print_mimic(sections_text):
    final_text = ''
    for section, text in sections_text.items():
        new_section = section.replace("_"," ").title()
        if section == 'admission_date':
            final_text += f"**{new_section}**: {text}"
        # elif section in ['discharge_date', 'sex']:
        #     final_text += f"**{new_section}**: {text}"
        else:
            final_text += f"\n\n**{new_section}**: {text}"
    
    return final_text