import json

class Vocab(object):
    def __init__(self):
        self.special_tokens = [ "<user_turn>", "<sys_turn>", "<turn_sep>", "<compare_sep>" ,"pricerange", "<pad>", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>",
                               "<eos_b>", "<eos_a>", "<go_d>",
                               "[restaurant]", "[hotel]", "[attraction]", "[src]", "[taxi]", "[police]", "[hospital]",
                               "[general]", "[inform]", "[request]",
                               "[nooffer]", "[recommend]", "[select]", "[offerbook]", "[offerbooked]", "[nobook]",
                               "[bye]", "[greet]", "[reqmore]", "[welcome]",
                               "[value_name]", "[value_choice]", "[value_area]", "[value_price]", "[value_type]",
                               "[value_reference]", "[value_phone]", "[value_address]",
                               "[value_food]", "[value_leave]", "[value_postcode]", "[value_id]", "[value_arrive]",
                               "[value_stars]", "[value_day]", "[value_destination]",
                               "[value_car]", "[value_departure]", "[value_time]", "[value_people]", "[value_stay]",
                               "[value_pricerange]", "[value_department]", "<None>", "[db_state0]", "[db_state1]",
                               "[db_state2]", "[db_state3]", "[db_state4]", "[db_state0+bookfail]",
                               "[db_state1+bookfail]", "[db_state2+bookfail]", "[db_state3+bookfail]",
                               "[db_state4+bookfail]", "[db_state0+booksuccess]", "[db_state1+booksuccess]",
                               "[db_state2+booksuccess]", "[db_state3+booksuccess]", "[db_state4+booksuccess]"]
        self.attr_special_tokens = {'pad_token': '<pad>',
                                    'additional_special_tokens': ["pricerange", "<go_r>", "<unk>", "<go_b>", "<go_a>",
                                                                  "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
                                                                  "[restaurant]", "[hotel]", "[attraction]", "[src]",
                                                                  "[taxi]", "[police]", "[hospital]", "[general]",
                                                                  "[inform]", "[request]",
                                                                  "[nooffer]", "[recommend]", "[select]", "[offerbook]",
                                                                  "[offerbooked]", "[nobook]", "[bye]", "[greet]",
                                                                  "[reqmore]", "[welcome]",
                                                                  "[value_name]", "[value_choice]", "[value_area]",
                                                                  "[value_price]", "[value_type]", "[value_reference]",
                                                                  "[value_phone]", "[value_address]",
                                                                  "[value_food]", "[value_leave]", "[value_postcode]",
                                                                  "[value_id]", "[value_arrive]", "[value_stars]",
                                                                  "[value_day]", "[value_destination]",
                                                                  "[value_car]", "[value_departure]", "[value_time]",
                                                                  "[value_people]", "[value_stay]",
                                                                  "[value_pricerange]", "[value_department]", "<None>",
                                                                  "[db_state0]", "[db_state1]", "[db_state2]",
                                                                  "[db_state3]", "[db_state4]", "[db_state0+bookfail]",
                                                                  "[db_state1+bookfail]", "[db_state2+bookfail]",
                                                                  "[db_state3+bookfail]", "[db_state4+bookfail]",
                                                                  "[db_state0+booksuccess]", "[db_state1+booksuccess]",
                                                                  "[db_state2+booksuccess]", "[db_state3+booksuccess]",
                                                                  "[db_state4+booksuccess]"]}

def load_damd_lex_delex_map(path="./data/multi-woz-processed/data_for_damd.json"):
    f = open(path)
    dataset = json.load(f)
    f.close()

    utt_lex_delex_map = {}

    for key in list(dataset.keys())[:]:
        dial = dataset[key]
        for utt in dial["log"]:
            user_delex = utt["user_delex"]
            user_lex = utt["user"]
            sys_delex = utt["resp"]
            sys_lex = utt["resp_nodelex"]
            utt_lex_delex_map[user_lex] = user_delex
            utt_lex_delex_map[sys_lex] = sys_delex

    return utt_lex_delex_map