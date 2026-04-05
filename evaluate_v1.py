import json
import os
import sys

# import subprocess

# def install(package):
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# install('pyvi')
# install('rouge')

# from rouge import Rouge
# from pyvi import ViTokenizer

ALL_ELEMENTS = ['subject', 'object', 'aspect', 'predicate']
ALL_LABELS = ['EQL', 'DIF', 'COM', 'COM+', 'COM-', 'SUP', 'SUP+', 'SUP-']

def evaluate_intra(gold_dir, pred_dir):
    def calculate_prf(ret, is_tuple=False):
        # Calculate precision, recall, and F1 for each matching criterion
        # only calculate proportional match for CEE, skip P if calculating for tuple
        for m in ['E', 'B'] if is_tuple else  ['E', 'P', 'B']:
            ret[f'{m}-P'] = ret[f'{m}-TP'] / (ret[f'{m}-TP'] + ret[f'{m}-FP']) if (ret[f'{m}-TP'] + ret[f'{m}-FP']) > 0 else 0.0
            ret[f'{m}-R'] = ret[f'{m}-TP'] / (ret[f'{m}-TP'] + ret[f'{m}-FN']) if (ret[f'{m}-TP'] + ret[f'{m}-FN']) > 0 else 0.0
            ret[f'{m}-F1'] = 2 * (ret[f'{m}-P'] * ret[f'{m}-R']) / (ret[f'{m}-P'] + ret[f'{m}-R']) if (ret[f'{m}-P'] + ret[f'{m}-R']) > 0 else 0.0

        return ret

    def get_and_add_infix(ret, infix):
        return {
            key.replace('-', f'-{infix}-'): ret[key] for key in ret
            if key.endswith('-P') or key.endswith('-R') or key.endswith('-F1')
        }

    def calculate_cee_metrics(sent_gold_tuples, sent_pred_tuples):
        def calculate_sentence_cee_metrics(gold_entities, pred_entities):
            def entity_to_set(entity):
                return set(int(k.split('&&')[0]) for k in entity)

            def exact_match(entity_set1, entity_set2):
                return int(entity_set1 == entity_set2)

            def proportional_match(entity_set1, entity_set2):
                intersection = entity_set1.intersection(entity_set2)
                return len(intersection) / len(entity_set1)

            def binary_match(entity_set1, entity_set2):
                return int(len(entity_set1.intersection(entity_set2)) != 0)

            preds = [entity_to_set(e) for e in pred_entities]
            golds = [entity_to_set(e) for e in gold_entities]

            ret = {
                'E-TP': 0, 'E-FP': 0, 'E-FN': 0, # exact match
                'P-TP': 0, 'P-FP': 0, 'P-FN': 0, # proportional match
                'B-TP': 0, 'B-FP': 0, 'B-FN': 0, # binary match
            }

            # calculate TP and FP
            for pred in preds:
                # exact match any entity in gold_entities
                if any(exact_match(pred, k) == 1 for k in golds):
                    ret['E-TP'] += 1
                else:
                    ret['E-FP'] += 1

                # proportional match with the longest matched entity in gold_entities
                max_match = max(proportional_match(pred, k) for k in golds) if len(golds) else 0
                ret['P-TP'] += max_match
                ret['P-FP'] += (1 - max_match)

                # binary match any entity in gold_entities
                if any(binary_match(pred, k) == 1 for k in golds):
                    ret['B-TP'] += 1
                else:
                    ret['B-FP'] += 1

            # calculate FN
            for gold in golds:
                # NOT exact match any entity in pred_entities
                if not any(exact_match(gold, k) == 1 for k in preds):
                    ret['E-FN'] += 1

                # proportional match FN is calculated on the the longest matched entity in pred_entities
                max_match = max(proportional_match(gold, k) for k in preds) if len(preds) else 0
                ret['P-FN'] += (1 - max_match)

                # NOT binary match any entity in pred_entities
                if not any(binary_match(gold, k) == 1 for k in preds):
                    ret['B-FN'] += 1

            return calculate_prf(ret)
        
        def calculate_cee_metrics_one_element(sent_gold_entities, sent_pred_entities):
            sentence_cee_metrics = [calculate_sentence_cee_metrics(g, p) for g, p in zip(sent_gold_entities, sent_pred_entities)]

            # calculate TP, FP, FN for all sentences
            ret = {
                k: sum(s_metrics[k] for s_metrics in sentence_cee_metrics)
                for k in [
                    'E-TP', 'E-FP', 'E-FN', # exact match
                    'P-TP', 'P-FP', 'P-FN', # proportional match
                    'B-TP', 'B-FP', 'B-FN', # binary match
                ]
            }

            return calculate_prf(ret)
        
        # get score for each element type S-subject~index 0, O-object~index 1, A-aspect~index 2, P-predicate~index 3
        element_results = [
            calculate_cee_metrics_one_element(
                [set(tuple(t[k]) for t in gold_tuples if len(t[k]) != 0) for gold_tuples in sent_gold_tuples],
                [set(tuple(t[k]) for t in pred_tuples if len(t[k]) != 0) for pred_tuples in sent_pred_tuples]
            ) for k in ALL_ELEMENTS
        ]
        
        ret = {
            **get_and_add_infix(element_results[0], 'CEE-S'),
            **get_and_add_infix(element_results[1], 'CEE-O'),
            **get_and_add_infix(element_results[2], 'CEE-A'),
            **get_and_add_infix(element_results[3], 'CEE-P'),
        }
        
        # calculate micro average P, R, F1
        ret.update(
            get_and_add_infix(
                calculate_prf({
                    k: sum(e_metrics[k] for e_metrics in element_results)
                    for k in [
                        'E-TP', 'E-FP', 'E-FN', # exact match
                        'P-TP', 'P-FP', 'P-FN', # proportional match
                        'B-TP', 'B-FP', 'B-FN', # binary match
                    ]
                }),
                'CEE-MICRO'
            )
        )
        
        # calculate macro average P, R, F1
        ret.update(
            get_and_add_infix(
                {
                    k: sum(e_metrics[k] for e_metrics in element_results)/4
                    for k in [
                        'E-P', 'E-R', 'E-F1', # exact match
                        'P-P', 'P-R', 'P-F1', # proportional match
                        'B-P', 'B-R', 'B-F1', # binary match
                    ]
                },
                'CEE-MACRO'
            )
        )

        return ret

    def calculate_tuple_metrics(sent_gold_tuples, sent_pred_tuples):
        def calculate_sentence_tuple_metrics(gold_tuples, pred_tuples, omit_label=True):
            def entity_to_set(entity):
                return set(int(k.split('&&')[0]) for k in entity)

            def tuple_to_set(tup):
                return {
                    **{
                        k: entity_to_set(tup[k]) for k in ALL_ELEMENTS
                    }, 'label': tup['label']
                }

            # all element should be exactly matched
            def exact_match(tup_set1, tup_set2):
                return int(
                    all(
                        tup_set1[k] == tup_set2[k]
                        for k in ALL_ELEMENTS
                    )
                    and (omit_label or tup_set1['label'] == tup_set2['label'])
                )

            # all element should be matched at least one token
            def binary_match(tup_set1, tup_set2):
                return int(
                    all(
                        len(tup_set1[k]) == len(tup_set2[k]) == 0 or len(tup_set1[k].intersection(tup_set2[k])) != 0
                        for k in ALL_ELEMENTS
                    )
                    and (omit_label or tup_set1['label'] == tup_set2['label'])
                )

                return int(len(entity_set1.intersection(entity_set2)) != 0)

            preds = [tuple_to_set(t) for t in pred_tuples]
            golds = [tuple_to_set(t) for t in gold_tuples]

            ret = {
                'E-TP': 0, 'E-FP': 0, 'E-FN': 0, # exact match
                'B-TP': 0, 'B-FP': 0, 'B-FN': 0, # binary match
            }

            # calculate TP and FP
            for pred in preds:
                # exact match any entity in gold_entities
                if any(exact_match(pred, k) == 1 for k in golds):
                    ret['E-TP'] += 1
                else:
                    ret['E-FP'] += 1

                # binary match any entity in gold_entities
                if any(binary_match(pred, k) == 1 for k in golds):
                    ret['B-TP'] += 1
                else:
                    ret['B-FP'] += 1

            # calculate FN
            for gold in golds:
                # NOT exact match any tuple in pred_tuples
                if not any(exact_match(gold, k) == 1 for k in preds):
                    ret['E-FN'] += 1

                # NOT binary match any tuple in pred_tuples
                if not any(binary_match(gold, k) == 1 for k in preds):
                    ret['B-FN'] += 1

            return calculate_prf(ret, is_tuple=True)

        def calculate_tuple_metrics_one_label(sent_gold_tuples, sent_pred_tuples, omit_label=True):
            sentence_tuple_metrics = [calculate_sentence_tuple_metrics(g, p, omit_label) for g, p in zip(sent_gold_tuples, sent_pred_tuples)]

            # calculate TP, FP, FN for all sentences
            ret = {
                k: sum(s_metrics[k] for s_metrics in sentence_tuple_metrics)
                for k in [
                    'E-TP', 'E-FP', 'E-FN', # exact match
                    'B-TP', 'B-FP', 'B-FN', # binary match
                ]
            }

            return calculate_prf(ret, is_tuple=True)
        
        def unique_tuple(list_of_dicts):
            def _str(tup):
                return f'{" ".join(tup["subject"])}_{" ".join(tup["object"])}_{" ".join(tup["aspect"])}_{" ".join(tup["predicate"])}_{tup["label"]}'

            unique_dicts = set()
            unique_dicts_list = []

            for d in list_of_dicts:
                s = _str(d)

                if s not in unique_dicts:
                    unique_dicts.add(s)
                    unique_dicts_list.append(d)

            return unique_dicts_list
        
        ret = {
            # tuple of four result
            **get_and_add_infix(calculate_tuple_metrics_one_label(sent_gold_tuples, sent_pred_tuples, omit_label=True), 'T4'),
        }

        # get score for each label
        label_results = [
            calculate_tuple_metrics_one_label(
                [unique_tuple(t for t in gold_tuples if t['label'] == k) for gold_tuples in sent_gold_tuples],
                [unique_tuple(t for t in pred_tuples if t['label'] == k) for pred_tuples in sent_pred_tuples],
                omit_label=False,
            ) for k in ALL_LABELS
        ]

        # add label results to final result
        for k, r in zip(ALL_LABELS, label_results):
            ret.update(
                get_and_add_infix(r, f'T5-{k}')
            )

        # calculate micro average P, R, F1
        ret.update(
            get_and_add_infix(
                calculate_prf({
                    k: sum(e_metrics[k] for e_metrics in label_results)
                    for k in [
                        'E-TP', 'E-FP', 'E-FN', # exact match
                        'B-TP', 'B-FP', 'B-FN', # binary match
                    ]
                }, is_tuple=True),
                'T5-MICRO'
            )
        )

        # calculate macro average P, R, F1
        ret.update(
            get_and_add_infix(
                {
                    k: sum(e_metrics[k] for e_metrics in label_results) / len(label_results)
                    for k in [
                        'E-P', 'E-R', 'E-F1', # exact match
                        'B-P', 'B-R', 'B-F1', # binary match
                    ]
                },
                'T5-MACRO'
            )
        )

        return ret

    def read_file(f_name):
        data = []
        with open(f_name, 'r') as f:
            sent_tuples = []
            txt = False
            for l in f:
                l = l.strip()

                if len(l) == 0:
                    if txt:
                        data.append(sent_tuples)
                    sent_tuples = []
                    txt = False
                elif l.startswith('{'):
                    sent_tuples.append(json.loads(l))
                else:
                    # text line
                    txt = True

            if txt:
                data.append(sent_tuples)

        return data

    def round_result(d):
        return {k: round(d[k], 4) for k in d}
        
    test_files = os.listdir(gold_dir)

    # Check if each file in the test_files exists
    for f_name in test_files:
        if not os.path.exists(os.path.join(pred_dir, f_name)):
            raise ValueError(f'The file "{f_name}" does not exist.')

    sent_gold_tuples = []
    sent_pred_tuples = []
    for f_name in test_files:
        gold_tuples = read_file(os.path.join(gold_dir, f_name))
        pred_tuples = read_file(os.path.join(pred_dir, f_name))

        if len(gold_tuples) != len(pred_tuples):
            raise ValueError(f'The file number of sentences in "{f_name}" does not match.')
        else:
            sent_gold_tuples.extend(gold_tuples)
            sent_pred_tuples.extend(pred_tuples)

    return round_result({
        **calculate_cee_metrics(sent_gold_tuples, sent_pred_tuples),
        **calculate_tuple_metrics(sent_gold_tuples, sent_pred_tuples)
    })
    

if __name__ == '__main__':
    [_, input_dir, output_dir] = sys.argv
    submission_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')

    result = evaluate_intra(truth_dir, submission_dir)

    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        output_file.write('E-T5-MACRO-F1: {:f}\n'.format(result['E-T5-MACRO-F1']))
        output_file.write('E-T5-MACRO-P: {:f}\n'.format(result['E-T5-MACRO-P']))
        output_file.write('E-T5-MACRO-R: {:f}\n'.format(result['E-T5-MACRO-R']))
        output_file.write('B-T5-MACRO-F1: {:f}\n'.format(result['B-T5-MACRO-F1']))
        output_file.write('B-T5-MACRO-P: {:f}\n'.format(result['B-T5-MACRO-P']))
        output_file.write('B-T5-MACRO-R: {:f}\n'.format(result['B-T5-MACRO-R']))
        output_file.write('E-T5-MICRO-F1: {:f}\n'.format(result['E-T5-MICRO-F1']))
        output_file.write('E-T5-MICRO-P: {:f}\n'.format(result['E-T5-MICRO-P']))
        output_file.write('E-T5-MICRO-R: {:f}\n'.format(result['E-T5-MICRO-R']))
        output_file.write('B-T5-MICRO-F1: {:f}\n'.format(result['B-T5-MICRO-F1']))
        output_file.write('B-T5-MICRO-P: {:f}\n'.format(result['B-T5-MICRO-P']))
        output_file.write('B-T5-MICRO-R: {:f}\n'.format(result['B-T5-MICRO-R']))
        output_file.write('E-T5-EQL-F1: {:f}\n'.format(result['E-T5-EQL-F1']))
        output_file.write('E-T5-EQL-P: {:f}\n'.format(result['E-T5-EQL-P']))
        output_file.write('E-T5-EQL-R: {:f}\n'.format(result['E-T5-EQL-R']))
        output_file.write('B-T5-EQL-F1: {:f}\n'.format(result['B-T5-EQL-F1']))
        output_file.write('B-T5-EQL-P: {:f}\n'.format(result['B-T5-EQL-P']))
        output_file.write('B-T5-EQL-R: {:f}\n'.format(result['B-T5-EQL-R']))
        output_file.write('E-T5-DIF-F1: {:f}\n'.format(result['E-T5-DIF-F1']))
        output_file.write('E-T5-DIF-P: {:f}\n'.format(result['E-T5-DIF-P']))
        output_file.write('E-T5-DIF-R: {:f}\n'.format(result['E-T5-DIF-R']))
        output_file.write('B-T5-DIF-F1: {:f}\n'.format(result['B-T5-DIF-F1']))
        output_file.write('B-T5-DIF-P: {:f}\n'.format(result['B-T5-DIF-P']))
        output_file.write('B-T5-DIF-R: {:f}\n'.format(result['B-T5-DIF-R']))
        output_file.write('E-T5-COM-F1: {:f}\n'.format(result['E-T5-COM-F1']))
        output_file.write('E-T5-COM-P: {:f}\n'.format(result['E-T5-COM-P']))
        output_file.write('E-T5-COM-R: {:f}\n'.format(result['E-T5-COM-R']))
        output_file.write('B-T5-COM-F1: {:f}\n'.format(result['B-T5-COM-F1']))
        output_file.write('B-T5-COM-P: {:f}\n'.format(result['B-T5-COM-P']))
        output_file.write('B-T5-COM-R: {:f}\n'.format(result['B-T5-COM-R']))
        output_file.write('E-T5-COM+-F1: {:f}\n'.format(result['E-T5-COM+-F1']))
        output_file.write('E-T5-COM+-P: {:f}\n'.format(result['E-T5-COM+-P']))
        output_file.write('E-T5-COM+-R: {:f}\n'.format(result['E-T5-COM+-R']))
        output_file.write('B-T5-COM+-F1: {:f}\n'.format(result['B-T5-COM+-F1']))
        output_file.write('B-T5-COM+-P: {:f}\n'.format(result['B-T5-COM+-P']))
        output_file.write('B-T5-COM+-R: {:f}\n'.format(result['B-T5-COM+-R']))
        output_file.write('E-T5-COM--F1: {:f}\n'.format(result['E-T5-COM--F1']))
        output_file.write('E-T5-COM--P: {:f}\n'.format(result['E-T5-COM--P']))
        output_file.write('E-T5-COM--R: {:f}\n'.format(result['E-T5-COM--R']))
        output_file.write('B-T5-COM--F1: {:f}\n'.format(result['B-T5-COM--F1']))
        output_file.write('B-T5-COM--P: {:f}\n'.format(result['B-T5-COM--P']))
        output_file.write('B-T5-COM--R: {:f}\n'.format(result['B-T5-COM--R']))
        output_file.write('E-T5-SUP-F1: {:f}\n'.format(result['E-T5-SUP-F1']))
        output_file.write('E-T5-SUP-P: {:f}\n'.format(result['E-T5-SUP-P']))
        output_file.write('E-T5-SUP-R: {:f}\n'.format(result['E-T5-SUP-R']))
        output_file.write('B-T5-SUP-F1: {:f}\n'.format(result['B-T5-SUP-F1']))
        output_file.write('B-T5-SUP-P: {:f}\n'.format(result['B-T5-SUP-P']))
        output_file.write('B-T5-SUP-R: {:f}\n'.format(result['B-T5-SUP-R']))
        output_file.write('E-T5-SUP+-F1: {:f}\n'.format(result['E-T5-SUP+-F1']))
        output_file.write('E-T5-SUP+-P: {:f}\n'.format(result['E-T5-SUP+-P']))
        output_file.write('E-T5-SUP+-R: {:f}\n'.format(result['E-T5-SUP+-R']))
        output_file.write('B-T5-SUP+-F1: {:f}\n'.format(result['B-T5-SUP+-F1']))
        output_file.write('B-T5-SUP+-P: {:f}\n'.format(result['B-T5-SUP+-P']))
        output_file.write('B-T5-SUP+-R: {:f}\n'.format(result['B-T5-SUP+-R']))
        output_file.write('E-T5-SUP--F1: {:f}\n'.format(result['E-T5-SUP--F1']))
        output_file.write('E-T5-SUP--P: {:f}\n'.format(result['E-T5-SUP--P']))
        output_file.write('E-T5-SUP--R: {:f}\n'.format(result['E-T5-SUP--R']))
        output_file.write('B-T5-SUP--F1: {:f}\n'.format(result['B-T5-SUP--F1']))
        output_file.write('B-T5-SUP--P: {:f}\n'.format(result['B-T5-SUP--P']))
        output_file.write('B-T5-SUP--R: {:f}\n'.format(result['B-T5-SUP--R']))
        output_file.write('E-T4-F1: {:f}\n'.format(result['E-T4-F1']))
        output_file.write('E-T4-P: {:f}\n'.format(result['E-T4-P']))
        output_file.write('E-T4-R: {:f}\n'.format(result['E-T4-R']))
        output_file.write('B-T4-F1: {:f}\n'.format(result['B-T4-F1']))
        output_file.write('B-T4-P: {:f}\n'.format(result['B-T4-P']))
        output_file.write('B-T4-R: {:f}\n'.format(result['B-T4-R']))
        output_file.write('E-CEE-MACRO-F1: {:f}\n'.format(result['E-CEE-MACRO-F1']))
        output_file.write('E-CEE-MACRO-P: {:f}\n'.format(result['E-CEE-MACRO-P']))
        output_file.write('E-CEE-MACRO-R: {:f}\n'.format(result['E-CEE-MACRO-R']))
        output_file.write('E-CEE-MICRO-F1: {:f}\n'.format(result['E-CEE-MICRO-F1']))
        output_file.write('E-CEE-MICRO-P: {:f}\n'.format(result['E-CEE-MICRO-P']))
        output_file.write('E-CEE-MICRO-R: {:f}\n'.format(result['E-CEE-MICRO-R']))
        output_file.write('E-CEE-S-F1: {:f}\n'.format(result['E-CEE-S-F1']))
        output_file.write('E-CEE-S-P: {:f}\n'.format(result['E-CEE-S-P']))
        output_file.write('E-CEE-S-R: {:f}\n'.format(result['E-CEE-S-R']))
        output_file.write('E-CEE-O-F1: {:f}\n'.format(result['E-CEE-O-F1']))
        output_file.write('E-CEE-O-P: {:f}\n'.format(result['E-CEE-O-P']))
        output_file.write('E-CEE-O-R: {:f}\n'.format(result['E-CEE-O-R']))
        output_file.write('E-CEE-A-F1: {:f}\n'.format(result['E-CEE-A-F1']))
        output_file.write('E-CEE-A-P: {:f}\n'.format(result['E-CEE-A-P']))
        output_file.write('E-CEE-A-R: {:f}\n'.format(result['E-CEE-A-R']))
        output_file.write('E-CEE-P-F1: {:f}\n'.format(result['E-CEE-P-F1']))
        output_file.write('E-CEE-P-P: {:f}\n'.format(result['E-CEE-P-P']))
        output_file.write('E-CEE-P-R: {:f}\n'.format(result['E-CEE-P-R']))
        output_file.write('P-CEE-MACRO-F1: {:f}\n'.format(result['P-CEE-MACRO-F1']))
        output_file.write('P-CEE-MACRO-P: {:f}\n'.format(result['P-CEE-MACRO-P']))
        output_file.write('P-CEE-MACRO-R: {:f}\n'.format(result['P-CEE-MACRO-R']))
        output_file.write('P-CEE-MICRO-F1: {:f}\n'.format(result['P-CEE-MICRO-F1']))
        output_file.write('P-CEE-MICRO-P: {:f}\n'.format(result['P-CEE-MICRO-P']))
        output_file.write('P-CEE-MICRO-R: {:f}\n'.format(result['P-CEE-MICRO-R']))
        output_file.write('P-CEE-S-F1: {:f}\n'.format(result['P-CEE-S-F1']))
        output_file.write('P-CEE-S-P: {:f}\n'.format(result['P-CEE-S-P']))
        output_file.write('P-CEE-S-R: {:f}\n'.format(result['P-CEE-S-R']))
        output_file.write('P-CEE-O-F1: {:f}\n'.format(result['P-CEE-O-F1']))
        output_file.write('P-CEE-O-P: {:f}\n'.format(result['P-CEE-O-P']))
        output_file.write('P-CEE-O-R: {:f}\n'.format(result['P-CEE-O-R']))
        output_file.write('P-CEE-A-F1: {:f}\n'.format(result['P-CEE-A-F1']))
        output_file.write('P-CEE-A-P: {:f}\n'.format(result['P-CEE-A-P']))
        output_file.write('P-CEE-A-R: {:f}\n'.format(result['P-CEE-A-R']))
        output_file.write('P-CEE-P-F1: {:f}\n'.format(result['P-CEE-P-F1']))
        output_file.write('P-CEE-P-P: {:f}\n'.format(result['P-CEE-P-P']))
        output_file.write('P-CEE-P-R: {:f}\n'.format(result['P-CEE-P-R']))
        output_file.write('B-CEE-MACRO-F1: {:f}\n'.format(result['B-CEE-MACRO-F1']))
        output_file.write('B-CEE-MACRO-P: {:f}\n'.format(result['B-CEE-MACRO-P']))
        output_file.write('B-CEE-MACRO-R: {:f}\n'.format(result['B-CEE-MACRO-R']))
        output_file.write('B-CEE-MICRO-F1: {:f}\n'.format(result['B-CEE-MICRO-F1']))
        output_file.write('B-CEE-MICRO-P: {:f}\n'.format(result['B-CEE-MICRO-P']))
        output_file.write('B-CEE-MICRO-R: {:f}\n'.format(result['B-CEE-MICRO-R']))
        output_file.write('B-CEE-S-F1: {:f}\n'.format(result['B-CEE-S-F1']))
        output_file.write('B-CEE-S-P: {:f}\n'.format(result['B-CEE-S-P']))
        output_file.write('B-CEE-S-R: {:f}\n'.format(result['B-CEE-S-R']))
        output_file.write('B-CEE-O-F1: {:f}\n'.format(result['B-CEE-O-F1']))
        output_file.write('B-CEE-O-P: {:f}\n'.format(result['B-CEE-O-P']))
        output_file.write('B-CEE-O-R: {:f}\n'.format(result['B-CEE-O-R']))
        output_file.write('B-CEE-A-F1: {:f}\n'.format(result['B-CEE-A-F1']))
        output_file.write('B-CEE-A-P: {:f}\n'.format(result['B-CEE-A-P']))
        output_file.write('B-CEE-A-R: {:f}\n'.format(result['B-CEE-A-R']))
        output_file.write('B-CEE-P-F1: {:f}\n'.format(result['B-CEE-P-F1']))
        output_file.write('B-CEE-P-P: {:f}\n'.format(result['B-CEE-P-P']))
        output_file.write('B-CEE-P-R: {:f}\n'.format(result['B-CEE-P-R']))
