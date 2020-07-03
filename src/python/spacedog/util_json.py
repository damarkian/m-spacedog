import json
import numpy as np

class NumpyEncoder(json.JSONEncoder): 

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dict2json(inputDict, jsonfile): 

    jdump = json.dumps(inputDict, cls=NumpyEncoder)

    with open(jsonfile, 'w') as f:
        json.dump(jdump, f)


