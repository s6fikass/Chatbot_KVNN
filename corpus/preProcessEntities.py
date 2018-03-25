import json

with open("kvret_train_public.json", 'r') as e:
    datastore = json.load(e)
    for dialogue in datastore:
        subject = None
        predicate = []
        for col in dialogue["scenario"]["kb"]["column_names"]:
            if subject is None:
                subject = col
            else:
                predicate.append(col)
        triples = {}
        i = 1
        if dialogue["scenario"]["kb"]["items"] is not None:
            for items in dialogue["scenario"]["kb"]["items"]:
                for pred in predicate:
                    if(pred in items):
                        triples[i] = [items[subject], pred, items[pred]]
                    else:
                        triples[i] = [items[subject], pred, "-"]
                    i = i+1
