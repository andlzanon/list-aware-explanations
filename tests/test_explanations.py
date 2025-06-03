import os
from pathlib import Path

import pandas as pd
import pytest
import re

from tqdm import tqdm


@pytest.mark.parametrize(
    "dataset, experiment",
    [
        ("hetrec2011-lastfm-2k", "experiment1"),
        ("ml-latest-small", "experiment1"),
        ("ml-latest-small", "experiment2"),
        ("ml-latest-small", "recsys_lbr_mobile1"),
        ("ml-latest-small", "recsys_lbr_mobile2"),
        ("ml-latest-small", "recsys_lbr_mobile3"),
        ("hetrec2011-lastfm-2k", "recsys_lbr_mobile1"),
        ("hetrec2011-lastfm-2k", "recsys_lbr_mobile2"),
        ("hetrec2011-lastfm-2k", "recsys_lbr_mobile3"),
    ]
)
def test_explanations(dataset: str, experiment: str):
    """
    Test to validate all samples explanations. An explanation to be validated must:
        (i) all interacted items should have all attributes shown on explanations
        (ii) all interacted items should have all attributes shown on explanations.
        Rule (ii) there is an exception for ExpLODRow algorithm when all_props=False. This means that
        the rule in this case is that at least one recommended item should have an attribute that appear on the
        recommendation
    have all attributes
    :param dataset: dataset to which explanations where generated
    :param experiment: explanation folder
    :return: True if all explanations for the experiment are valid, False otherwise
    """
    # get test dir and get the path to the explanations
    test_dir = Path(__file__).parent
    expl_folder = test_dir.parent / "datasets" / dataset / "stratified_split" / "explanations" / experiment

    # read KG
    if dataset == "ml-latest-small":
        prop_set = pd.read_csv(test_dir.parent / "knowledge-graphs" / "props_wikidata_movielens_small.csv",
                               usecols=["movieId", "title", "prop", "obj", "imdbId"])
    else:
        prop_set = pd.read_csv(test_dir.parent / "knowledge-graphs" / "props_artists_id.csv",
                               usecols=["id", "artist", "prop", "obj", "name"])

    # index by item name (column 1) that is used on explanations
    prop_set = prop_set.set_index(prop_set.columns[1])

    # for every algorithm on the experiment
    for filename in os.listdir(expl_folder):
        filepath = os.path.join(expl_folder, filename)

        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                n_lines = len(file.readlines())

            with open(filepath, 'r', encoding='utf-8') as file:
                print(filepath.split("\\")[-1] + " | " + dataset + " | " + experiment)
                # parse every line and get attributes (objs), hist items and rec items
                for line in tqdm(file, total=n_lines, desc="Validating explanations on file..."):
                    ret = parse_sentence(line.rstrip(), prop_set)
                    if sum([len(r) for r in ret]) > 0 and len(ret[0]) > 0:
                        objs = ret[0]
                        hist = ret[1]
                        rec = ret[2]
                        h_ids = []
                        # check if all hist items have all obj
                        for h in hist:
                            h_id = prop_set.loc[h][prop_set.columns[-1]].unique()[0]
                            if h_id not in h_ids:
                                h_ids.append(h_id)
                                try:
                                    h_loc = list(prop_set[prop_set[prop_set.columns[-1]] == h_id].index.unique())
                                    df_h = prop_set.loc[h_loc]
                                except KeyError:
                                    print("Did not find item " + h + " in prop_set")
                                for obj in objs:
                                    if df_h[df_h['obj'] == obj].shape[0] == 0:
                                        print(line.rstrip())
                                        assert False

                        # check if all rec have attribute if file is not ExpLODRows with all_props param = False
                        if "all_props=False" not in str(filepath):
                            r_ids = []
                            for r in rec:
                                r_id = prop_set.loc[r][prop_set.columns[-1]].unique()[0]
                                if r_id not in r_ids:
                                    r_ids.append(r_id)
                                    try:
                                        r_loc = list(prop_set[prop_set[prop_set.columns[-1]] == r_id].index.unique())
                                        df_r = prop_set.loc[r_loc]
                                    except KeyError:
                                        print("Did not find item " + r + " in prop_set")
                                    for obj in objs:
                                        if df_r[df_r['obj'] == obj].shape[0] == 0:
                                            print(line.rstrip())
                                            assert False
                        # Otherwise check if at least one have the obj
                        else:
                            for obj in objs:
                                if len(set(prop_set[prop_set['obj'] == obj].index).intersection(set(rec))) == 0:
                                    print(line.rstrip())
                                    assert False

    assert True

def smart_split(text, values: list):
    """
    Split the line and treat exceptions
    :param text: string of terms separated by comma. E.g.: Titanic, Forrest Gump
    :param values: string of values (properties or movies)
    :return: splited text: [Titanic, Forrest Gump]
    """
    parts = re.split(', ', text)
    merged = []
    i = 0
    while i < len(parts):
        current = parts[i].strip()
        if current in values:
            if current.lower() not in (item.lower() for item in merged):
                merged.append(current)
            i = i + 1
        else:
            try:
                current = parts[i] + ', ' + parts[i+1]
                if current in values:
                    merged.append(current)
                    i = i + 2
                else:
                    current = parts[i - 1] + ', ' + parts[i]
                    if current in values:
                        merged[len(merged) - 1] = current
                        i = i + 1
                    else:
                        found = False
                        c = 2
                        j = i + c
                        current = parts[i] + ', ' + parts[i+1]
                        while j < len(parts) and not found:
                            current = current + ', ' + parts[j]
                            if current in values:
                                merged.append(current)
                                c = c + 1
                                i = i + c
                                found = True
                            else:
                                c = c + 1
                                j = i + c

                        if not found:
                            c = 1
                            j = i + c
                            current = parts[i - 1] + ', ' + parts[i]
                            while j < len(parts) and not found:
                                current = current + ', ' + parts[j]
                                if current in values:
                                    merged[len(merged) - 1] = current
                                    c = c + 1
                                    i = i + c
                                    found = True
                                else:
                                    c = c + 1
                                    j = i + c

                        if not found:
                            raise ValueError(f'''Value {current} does not exist on prop_set''')

            except IndexError:
                current = parts[i-1] + ', ' + parts[i]
                if current in values:
                    merged[len(merged)-1] = current
                    i = i + 1
                else:
                    raise IndexError

    return merged

def parse_sentence(sentence: str, prop_set: pd.DataFrame):
    """
    Parsing function
    "If you are in the mood for musician, United States of America items, I recommend Green Day, The Doors"
    where we get the musician and United States of America, on a list and Green Day and The Doors on another.
    Another case is when there "such as" clauses. E.g.: "If you are in the mood for electronic rock items such
    as Matryoshka, I recommend Placebo, Depeche Mode, Daft Punk"
    Code generated by ChatGPT and changed by me.
    :param prop_set: knowledge graph
    :param sentence: phrase to parse
    :return: tuple with the criteria, examples, recommendations
    """
    props = list(prop_set['obj'].unique())
    items = list(prop_set.index.unique().astype(str))

    # 1. Match case: no criteria, just examples and recommendations
    match_no_criteria = re.search(
        r"mood for items, items such as (.+?), I recommend (.+)", sentence
    )
    if match_no_criteria:
        examples_str = match_no_criteria.group(1)
        recs_str = match_no_criteria.group(2)

        return [], smart_split(examples_str, items), smart_split(recs_str, items)

    # 2. Match case: criteria + examples + recommendations
    match_with_examples = re.search(
        r"mood for (.+?) items such as (.+?), I recommend (.+)", sentence
    )
    if match_with_examples:
        criteria_str = match_with_examples.group(1)
        examples_str = match_with_examples.group(2)
        recs_str = match_with_examples.group(3)

        return (
            smart_split(criteria_str, props),
            smart_split(examples_str, items),
            smart_split(recs_str, items)
        )

    # 3. Match case: criteria + recommendations only
    match_basic = re.search(r"mood for (.+?) items, I recommend (.+)", sentence)
    if match_basic:
        criteria_str = match_basic.group(1)
        recs_str = match_basic.group(2)

        return (
            smart_split(criteria_str, props),
            [],
            smart_split(recs_str, items)
        )

    # Fallback
    return [], [], []

