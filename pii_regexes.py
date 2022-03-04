import regex_manager
import pii_regexes_rulebase

#For backwards compatability. Everything moved to regex_manager.
regex_rulebase = pii_regexes_rulebase.regex_rulebase
detect_ner_with_regex_and_context = regex_manager.detect_ner_with_regex_and_context

