import os

import numpy as np
import pandas as pd

entity_names = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                        "data", "raw", "smap_msl", "labeled_anomalies.csv"))[["chan_id", "spacecraft"]]
#
# entity_names = pd.DataFrame(columns=["chan_id", "spacecraft"])
# entity_names["chan_id"] = ['P-1', 'S-1', 'E-1', 'E-2', 'E-3', 'E-4', 'E-5', 'E-6', 'E-7',
#        'E-8', 'E-9', 'E-10', 'E-11', 'E-12', 'E-13', 'A-1', 'D-1', 'P-2',
#        'P-3', 'D-2', 'D-3', 'D-4', 'A-2', 'A-3', 'A-4', 'G-1', 'G-2',
#        'D-5', 'D-6', 'D-7', 'F-1', 'P-4', 'G-3', 'T-1', 'T-2', 'D-8',
#        'D-9', 'F-2', 'G-4', 'T-3', 'D-11', 'D-12', 'B-1', 'G-6', 'G-7',
#        'P-7', 'R-1', 'A-5', 'A-6', 'A-7', 'D-13', 'P-2', 'A-8', 'A-9',
#        'F-3', 'M-6', 'M-1', 'M-2', 'S-2', 'P-10', 'T-4', 'T-5', 'F-7',
#        'M-3', 'M-4', 'M-5', 'P-15', 'C-1', 'C-2', 'T-12', 'T-13', 'F-4',
#        'F-5', 'D-14', 'T-9', 'P-14', 'T-8', 'P-11', 'D-15', 'D-16', 'M-7',
#        'F-8', '=======', 'chan_id', 'P-1', 'S-1', 'E-1', 'E-2', 'E-3',
#        'E-4', 'E-5', 'E-6', 'E-7', 'E-8', 'E-9', 'E-10', 'E-11', 'E-12',
#        'E-13', 'A-1', 'D-1', 'P-2', 'P-3', 'D-2', 'D-3', 'D-4', 'A-2',
#        'A-3', 'A-4', 'G-1', 'G-2', 'D-5', 'D-6', 'D-7', 'F-1', 'P-4',
#        'G-3', 'T-1', 'T-2', 'D-8', 'D-9', 'F-2', 'G-4', 'T-3', 'D-11',
#        'D-12', 'B-1', 'G-6', 'G-7', 'P-7', 'R-1', 'A-5', 'A-6', 'A-7',
#        'D-13', 'P-2', 'A-8', 'A-9', 'F-3', 'M-6', 'M-1', 'M-2', 'S-2',
#        'P-10', 'T-4', 'T-5', 'F-7', 'M-3', 'M-4', 'M-5', 'P-15', 'C-1',
#        'C-2', 'T-12', 'T-13', 'F-4', 'F-5', 'D-14', 'T-9', 'P-14', 'T-8',
#        'P-11', 'D-15', 'D-16', 'M-7', 'F-8']
#
# entity_names["spacecraft"] = ['SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'MSL',
#        'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL',
#        'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL',
#        'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', None,
#        'spacecraft', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP', 'SMAP',
#        'SMAP', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL',
#        'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL',
#        'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL', 'MSL',
#        'MSL']

smap_entities = sorted(np.unique(entity_names[entity_names["spacecraft"] == "SMAP"]["chan_id"].values.astype(str)))
msl_entities = sorted(np.unique(entity_names[entity_names["spacecraft"] == "MSL"]["chan_id"].values.astype(str)))
smd_entities = sorted(["machine-1-%i" % i for i in range(1, 9)] + ["machine-2-%i" % i for i in range(1, 10)] +
                      ["machine-3-%i" % i for i in range(1, 12)])
entities_dict = {"smd": smd_entities, "smap": smap_entities, "msl": msl_entities}