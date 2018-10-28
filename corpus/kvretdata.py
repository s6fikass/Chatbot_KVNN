

import os
import ast
import json


"""
Load the Kvert data .

Available from here:
https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/

"""


class KvretData:
    """

    """

    def __init__(self, fileName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        self.lines = {}
        self.conversations = []

        LINE_FIELDS = ["lineID", "turn", "text"]

        INTENT ={
            "navigate": "poi",
            "schedule": "event",
            "weather": "location"
        }

        CALENDAR_COLUMN_NAMES = ["event", "time", "date", "room", "agenda", "party"]

        lOCATION_INFORMATION_COLUMN_NAMES = ["poi", "poi_type", "address", "distance", "traffic_info"]

        WEATHER_COLUMN_NAMES = ["location", "monday", "tuesday", "wednesday", "thursday", "friday",
                                "saturday", "sunday", "today"]

        CONVERSATIONS_FIELDS = ["utteranceIDs"]
        #
       # dataStores = PreProcess(os.path.join(dirName,'kvret_train_public.json'), os.path.join(dirName,'kvret_test_public.json'))

        [self.lines, self.conversations] = self.loadLines(fileName)
        # self.conversations = self.loadConversations(os.path.join(dirName, "kvret_train_conversations.txt"),
        #                                             CONVERSATIONS_FIELDS)
        # print self.conversations[1]

    def loadLines(self, fileName):
        """
        Args:
            fileName (str): file to load
            field (set<str>): fields to extract
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        conversation = []
        lines = {}
        conversationId = 0
        lineID = 1
        print(fileName)
        with open(fileName, 'r') as f:  # TODO: Solve Iso encoding pb !
            datastore = json.load(f)
            for dialogue in datastore:
                convObj = {}
                conversationId = conversationId + 1
                convObj["lines"] = []
                for utterence in dialogue["dialogue"]:
                    lineID = lineID + 1
                    lineObj = {}
                    lineObj['turn'] = utterence['turn']
                    lineObj['utterance'] = utterence['data']['utterance']
                    if lineObj['turn'] == 'assistant':
                        requested = []
                        for knowledgeRequested in utterence['data']['requested']:
                            if utterence['data']['requested'][knowledgeRequested]:
                                requested.append(knowledgeRequested)
                        lineObj["requested"] = requested
                        lineObj["slots"] = utterence['data']['slots']
                    lines[lineID] = lineObj
                    convObj["lines"].append(lineObj)
                # EOS
                convObj[conversationId] = conversationId

                # Get KB entries
                predicate = []
                subject = None
                for col in dialogue["scenario"]["kb"]["column_names"]:
                    if subject is None:
                        subject = col
                    else:
                        predicate.append(col)
                triples = []
                if dialogue["scenario"]["kb"]["items"] is not None:
                    for items in dialogue["scenario"]["kb"]["items"]:
                        for pred in predicate:
                            if (pred in items):
                                triples.append([items[subject], pred, items[pred]])
                            else:
                                triples.append([items[subject], pred, "-"])
                convObj["kb"] = triples
                convObj["intent"] = dialogue["scenario"]["task"]["intent"]
                conversation.append(convObj)
        return [lines, conversation]

    def loadConversations(self, fileName, fields):
        """
        Args:
            fileName (str): file to load
            field (set<str>): fields to extract
        Return:
            dict<dict<str>>: the extracted fields for each line
        """
        conversations = []

        with open(fileName, 'r') as f:  # TODO: Solve Iso encoding pb !
            for line in f:
                values = line.split(" +++$+++ ")

                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]

                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                lineIds = ast.literal_eval(convObj["utteranceIDs"])

                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId])

                conversations.append(convObj)

        return conversations

    def loadLinesFromJson(self, jsonfile):
        return ""

    def getConversations(self):
        return self.conversations
