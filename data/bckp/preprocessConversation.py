
import json
filename = '../data/kvret_train_public'
Mode='_train_'
lineID = 0
conversationId = 0
kb = {}
datastore = ''
with open(filename + '.json', 'r') as f:
    datastore = json.load(f)

# Use the new datastore datastructure
for dialogue in datastore:
    conversationId = conversationId + 1
    conversationIds = []
    for utterence in dialogue["dialogue"]:
        if utterence['turn'] == 'driver':
            with open(filename + Mode + 'input.txt', 'a') as f:
                lineID = lineID + 1
                conversationIds.append("'L{0}'".format(lineID))
                f.write(utterence['data']['utterance'] + '\n')
                f.close()
        else:
            with open(filename + Mode + 'target.txt', 'a') as f:
                lineID = lineID + 1
                conversationIds.append("'L{0}'".format(lineID))
                f.write(utterence['data']['utterance'] + '\n')
                f.close()
    with open('{0}_conversations.txt'.format(filename), 'a') as c:
        c.write('C{0}'.format(conversationId) + ' +++$+++ ' + '[' + ', '.join(conversationIds) + ']')
        c.write('\n')
        c.close()
    kb['C{0}'.format(conversationId)] = dialogue['scenario']['kb']['items']
with open('{0}_kb_items.json'.format(filename), 'a') as kbfile:
    json.dump(kb, kbfile)