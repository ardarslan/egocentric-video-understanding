import os
from nltk.parse.corenlp import CoreNLPServer


server = CoreNLPServer(
    port=5960,
    path_to_jar=os.path.join(
        os.environ["SCRATCH"],
        "mq_libs/stanford-corenlp",
        "stanford-corenlp-4.5.5.jar",
    ),
    path_to_models_jar=os.path.join(
        os.environ["SCRATCH"],
        "mq_libs/stanford-corenlp",
        "stanford-corenlp-4.5.5-models.jar",
    ),
)
server.start()
