# Chatbot_KVNN

to run the model in training mode:

```
python3 pytourch_main.py
```

To evaluate a certain model :

```
python3 pytourch_main.py -tm < model path > -vv 1
```

Every update in textdata.py needs reprocessing and deleting files in ```data/samples```