import re
# TODO, copy in the code from https://github.com/bigscience-workshop/data_tooling/blob/master/ac_dc/anonymization.py
rulebase = {"en": [([
                        ("AGE", re.compile("\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old"), None, None,
                         None),
                        ("STREET_ADDRESS", re.compile(
                            '\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)'),
                         None, None, None),
                        ("STREET_ADDRESS", re.compile('P\.? ?O\.? Box \d+'), None, None, None),
                        ("GOVT_ID", re.compile(
                            '(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}'),
                         None, None, None),
                        ("DISEASE", re.compile("diabetes|cancer|HIV|AIDS|Alzheimer's|Alzheimer|heart disease"), None,
                         None, None),
                        ("NORP", re.compile("upper class|middle class|working class|lower class"), None, None, None),
                    ], 1),
],
    "vi": []
}




