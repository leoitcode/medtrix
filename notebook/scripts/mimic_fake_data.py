import random
import re
import pickle


with open('list_fake_data.pkl', 'rb') as f:
    list_fake_data = pickle.load(f)

class PlaceholderMapper:

    def __init__(self, lists_replacements):

        self.placeholder_mapping = {}
        self.lists_replacements = list_fake_data

    def get_mapping(self, placeholder):

        mo = re.match("\[\*\*Age over 90 \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(90, 100))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Age over 90 \*\*\]", placeholder)
        if mo:
            return str(random.randint(90, 100))

        mo = re.match("\[\*\*Apartment Address\(\d+\) \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["addresses"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Apartment Address\(\d+\) \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["addresses"])

        mo = re.match("\[\*\*Attending Info \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                firstname = random.choice(self.lists_replacements["all_first_names"])
                name = random.choice(self.lists_replacements["last_names"])
                self.placeholder_mapping[mo.group(0)] = "{} {}".format(firstname, name)
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Attending Info \*\*\]", placeholder)
        if mo:
            firstname = random.choice(self.lists_replacements["all_first_names"])
            name = random.choice(self.lists_replacements["last_names"])
            return "{} {}".format(firstname, name)

        mo = re.match("\[\*\*CC Contact Info \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["phone_numbers"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*CC Contact Info \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["phone_numbers"])

        mo = re.match("\[\*\*Clip Number \(Radiology\) \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1, 10000))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Clip Number \(Radiology\) \*\*\]", placeholder)
        if mo:
            return str(random.randint(1, 10000))

        mo = re.match("\[\*\*Company \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["companies"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Company \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["companies"])

        mo = re.match("\[\*\*Country \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["countries"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Country \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["countries"])

        mo = re.match("\[\*\*Date (r|R)ange (\(\d+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                year_begin, month_begin, day_begin, year_end, month_end, day_end = self._build_date_range()
                self.placeholder_mapping[mo.group(0)] = "{}/{}/{}-{}/{}/{}".format(
                    year_begin,
                    month_begin,
                    day_begin,
                    year_end,
                    month_end,
                    day_end
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Date (r|R)ange (\(\d+\) )?\*\*\]", placeholder)
        if mo:
            year_begin, month_begin, day_begin, year_end, month_end, day_end = self._build_date_range()
            return "{}/{}/{}-{}/{}/{}".format(
                year_begin,
                month_begin,
                day_begin,
                year_end,
                month_end,
                day_end
            )

        mo = re.match("\[\*\*Dictator Info \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1, 10000))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Dictator Info \*\*\]", placeholder)
        if mo:
            return str(random.randint(1, 10000))

        mo = re.match("\[\*\*Doctor First Name \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["all_first_names"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Doctor First Name \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["all_first_names"])

        mo = re.match("\[\*\*Doctor Last Name (\(ambig\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["last_names"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Doctor Last Name (\(ambig\) )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["last_names"])

        mo = re.match("\[\*\*E-mail address \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["emails"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*E-mail address \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["emails"])

        mo = re.match("\[\*\*Female First Name \([^\[]+\) \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["first_names_female"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Female First Name \([^\[]+\) \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["first_names_female"])

        mo = re.match("\[\*\*First Name(\d+)? (\([^\[]+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["all_first_names"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*First Name(\d+)? (\([^\[]+\) )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["all_first_names"])

        mo = re.match("\[\*\*Holiday \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["holidays"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Holiday \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["holidays"])

        mo = re.match("\[\*\*Hospital(\d+)? \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["hospitals"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Hospital(\d+)? \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["hospitals"])

        mo = re.match("\[\*\*Initials? \(NamePattern\d+\) \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                firstname = random.choice(self.lists_replacements["all_first_names"])
                name = random.choice(self.lists_replacements["last_names"])
                self.placeholder_mapping[mo.group(0)] = "{}{}".format(firstname[0:1], name[0:1])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Initials? \(NamePattern\d+\) \*\*\]", placeholder)
        if mo:
            firstname = random.choice(self.lists_replacements["all_first_names"])
            name = random.choice(self.lists_replacements["last_names"])
            return "{}{}".format(firstname[0:1], name[0:1])

        mo = re.match("\[\*\*Job Number \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1, 10000))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Job Number \*\*\]", placeholder)
        if mo:
            return str(random.randint(1, 10000))

        mo = re.match("\[\*\*Known firstname \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["all_first_names"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Known firstname \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["all_first_names"])

        mo = re.match("\[\*\*Known lastname \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["last_names"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Known lastname \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["last_names"])

        mo = re.match("\[\*\*Last Name ([^\[]+ )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["last_names"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Last Name ([^\[]+ )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["last_names"])

        mo = re.match("\[\*\*Location ([^\[]+ )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["locations"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Location ([^\[]+ )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["locations"])

        mo = re.match("\[\*\*MD Number(\(\d+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["phone_numbers"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*MD Number(\(\d+\) )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["phone_numbers"])

        mo = re.match("\[\*\*Male First Name (\([^[]+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["all_first_names"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Male First Name (\([^[]+\) )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["all_first_names"])

        mo = re.match("\[\*\*Medical Record Number (\([^[]+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1, 10000))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Medical Record Number (\([^[]+\) )?\*\*\]", placeholder)
        if mo:
            return str(random.randint(1, 10000))

        mo = re.match("\[\*\*Month \(only\) \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["months"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Month \(only\) \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["months"])

        mo = re.match("\[\*\*Month Day \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1, 31))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Month Day \*\*\]", placeholder)
        if mo:
            return str(random.randint(1, 31))

        mo = re.match("\[\*\*Month/Day (\(?\d+\)? )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}/{}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1, 31))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Month/Day (\(?\d+\)? )?\*\*\]", placeholder)
        if mo:
            return "{}/{}".format(
                str(random.randint(1, 12)),
                str(random.randint(1, 31))
            )

        mo = re.match("\[\*\*Month/Year (\(?\d+\)? )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}/{}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1950, 2016))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Month/Year (\(?\d+\)? )?\*\*\]", placeholder)
        if mo:
            return "{}/{}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1950, 2016))
                )

        mo = re.match("\[\*\*Month/Day/Year \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}/{}/{}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1, 31)),
                    str(random.randint(1950, 2016))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Month/Day/Year \*\*\]", placeholder)
        if mo:
            return "{}/{}/{}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1, 31)),
                    str(random.randint(1950, 2016))
                )

        mo = re.match("\[\*\*Name(\d+)? (\([^\[]+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["last_names"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Name(\d+)? (\([^\[]+\) )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["last_names"])

        mo = re.match("\[\*\*Name Initial (\([^\[]*\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                firstname = random.choice(self.lists_replacements["all_first_names"])
                name = random.choice(self.lists_replacements["last_names"])
                self.placeholder_mapping[mo.group(0)] = "{}{}".format(firstname[0:1], name[0:1])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Name Initial (\([^\[]*\) )?\*\*\]", placeholder)
        if mo:
            firstname = random.choice(self.lists_replacements["all_first_names"])
            name = random.choice(self.lists_replacements["last_names"])
            return "{}{}".format(firstname[0:1], name[0:1])

        mo = re.match("\[\*\*Numeric Identifier \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1, 10000))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Numeric Identifier \*\*\]", placeholder)
        if mo:
            return str(random.randint(1, 10000))

        mo = re.match("\[\*\*Pager number \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["phone_numbers"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Pager number \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["phone_numbers"])

        mo = re.match("\[\*\*Provider Number \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["phone_numbers"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Provider Number \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["phone_numbers"])

        mo = re.match("\[\*\*Serial Number \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}-{}-{}".format(
                    str(random.randint(1, 10000)),
                    str(random.randint(1, 10000)),
                    str(random.randint(1, 10000))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Serial Number \*\*\]", placeholder)
        if mo:
            return "{}-{}-{}".format(
                    str(random.randint(1, 10000)),
                    str(random.randint(1, 10000)),
                    str(random.randint(1, 10000))
                )

        mo = re.match("\[\*\*Social Security Number \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["ssn"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Social Security Number \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["ssn"])

        mo = re.match("\[\*\*State \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["states"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*State \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["states"])

        mo = re.match("\[\*\*Street Address(\(\d+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["addresses"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Street Address(\(\d+\) )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["addresses"])

        mo = re.match("\[\*\*Telephone/Fax (\(\d+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["phone_numbers"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Telephone/Fax (\(\d+\) )?\*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["phone_numbers"])

        mo = re.match("\[\*\*Unit Number \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1, 10000))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Unit Number \*\*\]", placeholder)
        if mo:
            return str(random.randint(1, 10000))

        mo = re.match("\[\*\*(\d\d\d\d-\d?\d-\d?\d)\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = mo.group(1)
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Year \((\d+) digits\) \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1950, 2016))[int(mo.group(1)):]
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Year \((\d+) digits\) \*\*\]", placeholder)
        if mo:
            return str(random.randint(1950, 2016))[int(mo.group(1)):]

        mo = re.match("\[\*\*Year/Month/Day \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}/{}/{}".format(
                    str(random.randint(1950, 2016)),
                    str(random.randint(1, 12)),
                    str(random.randint(1, 31))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Year/Month/Day \*\*\]", placeholder)
        if mo:
            return "{}/{}/{}".format(
                    str(random.randint(1950, 2016)),
                    str(random.randint(1, 12)),
                    str(random.randint(1, 31))
                )

        mo = re.match("\[\*\*((January|February|March|April|May|June|July|August|"
                      "September|October|November|December) \d+)\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = mo.group(1)
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Name Prefix \(Prefixes\) \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(['Ms', 'Miss', 'Mrs', 'Mr', 'Dr', 'Prof'])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Name Prefix \(Prefixes\) \*\*\]", placeholder)
        if mo:
            return random.choice(['Ms', 'Miss', 'Mrs', 'Mr', 'Dr', 'Prof'])

        mo = re.match("\[\*\*PO Box \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "PO BOX {}".format(
                    str(random.randint(1, 1000))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*PO Box \*\*\]", placeholder)
        if mo:
            return "PO BOX {}".format(
                    str(random.randint(1, 1000))
                )

        mo = re.match("\[\*\*Year/Month \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}/{}".format(
                    str(random.randint(1950, 2016)),
                    str(random.randint(1, 12))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Year/Month \*\*\]", placeholder)
        if mo:
            return "{}/{}".format(
                    str(random.randint(1950, 2016)),
                    str(random.randint(1, 12))
                )

        mo = re.match("\[\*\*Month Day Year (\(\d+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{} {} {}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1, 31)),
                    str(random.randint(1950, 2016))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Month Day Year (\(\d+\) )?\*\*\]", placeholder)
        if mo:
            return "{} {} {}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1, 31)),
                    str(random.randint(1950, 2016))
                )

        mo = re.match("\[\*\*Month Year \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{} {}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1950, 2016))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Month Year \*\*\]", placeholder)
        if mo:
            return "{} {}".format(
                    str(random.randint(1, 12)),
                    str(random.randint(1950, 2016))
                )

        mo = re.match("\[\*\*Day Month \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{} {}".format(
                    str(random.randint(1, 31)),
                    str(random.randint(1, 12))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Day Month \*\*\]", placeholder)
        if mo:
            return "{} {}".format(
                    str(random.randint(1, 31)),
                    str(random.randint(1, 12))
                )

        mo = re.match("\[\*\*Day Month Year (\(\d+\) )?\d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{} {} {}".format(
                    str(random.randint(1, 31)),
                    str(random.randint(1, 12)),
                    str(random.randint(1950, 2016))
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Day Month Year (\(\d+\) )?\*\*\]", placeholder)
        if mo:
            return "{} {} {}".format(
                    str(random.randint(1, 31)),
                    str(random.randint(1, 12)),
                    str(random.randint(1950, 2016))
                )

        mo = re.match("\[\*\*State/Zipcode \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = str(random.randint(1, 99999))
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*State/Zipcode \*\*\]", placeholder)
        if mo:
            return str(random.randint(1, 99999))

        mo = re.match("\[\*\*Hospital Unit Number \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["phone_numbers"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Hospital Unit Number \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["phone_numbers"])

        mo = re.match("\[\*\*University/College \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["colleges"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*University/College \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["colleges"])

        mo = re.match("\[\*\*Hospital Ward Name \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["wards_units"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Hospital Ward Name \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["wards_units"])

        mo = re.match("\[\*\*Hospital Unit Name \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["wards_units"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Hospital Unit Name \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["wards_units"])

        mo = re.match("\[\*\*Wardname \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["wards_units"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*Wardname \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["wards_units"])

        mo = re.match("\[\*\*URL \d+\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = random.choice(self.lists_replacements["websites"])
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*URL \*\*\]", placeholder)
        if mo:
            return random.choice(self.lists_replacements["websites"])

        mo = re.match("\[\*\* \d+\*\*\]", placeholder)
        if mo:
            return ''

        mo = re.match("\[\*\*\s\*\*\]", placeholder)
        if mo:
            return ''

        mo = re.match("\[\*\*(\d+)-/(\d+)\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}/{}".format(
                    mo.group(1),
                    mo.group(2)
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*(\d+)/(\d+)\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}/{}".format(
                    mo.group(1),
                    mo.group(2)
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*(\d+)-(\d+)\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}-{}".format(
                    mo.group(1),
                    mo.group(2)
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*-(\d+)/(\d+)\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = "{}/{}".format(
                    mo.group(1),
                    mo.group(2)
                )
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*(\d+-\d+-\d+)\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = mo.group(1)
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*(\d+)\*\*\]", placeholder)
        if mo:
            if mo.group(0) not in self.placeholder_mapping:
                self.placeholder_mapping[mo.group(0)] = mo.group(1)
            return self.placeholder_mapping[mo.group(0)]

        mo = re.match("\[\*\*[^\[]*\*\*\]", placeholder)
        if mo:
            return ''
            # return placeholder

    @staticmethod
    def _build_date_range():

        year_begin = random.randint(1950, 2016)
        month_begin = random.randint(1, 12)
        day_begin = random.randint(1, 28)

        year_end = random.randint(year_begin, year_begin + 2)
        if year_end > year_begin:
            month_end = random.randint(1, 12)
            day_end = random.randint(1, 28)
        else:
            month_end = random.randint(month_begin, 12)
            if month_end > month_begin:
                day_end = random.randint(1, 28)
            else:
                day_end = random.randint(day_begin, 28)

        return year_begin, month_begin, day_begin, year_end, month_end, day_end