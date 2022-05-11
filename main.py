import os
import argparse
import config

from stardist.models import StarDist2D

import prediction

parser = argparse.ArgumentParser(description="""
    using this script to auto segment the cell images and identify each cell's cell cycle phase.
    usage:
        python main.py -pcna <your pcna image filepath> [optional] -bf <your bf image filepath> -o <output result filepath> 
""")

parser.add_argument('-pcna', default=False, help="input image filepath of pcna")
parser.add_argument('-o', default=False, help='output json file path')
parser.add_argument('-bf', default=False, help='input image filepath of bright field')

args = parser.parse_args()

if args.pcna is False:
    raise ValueError("pcna image must be given!")
else:
    pcna = args.pcna
if args.bf is False:
    # warnings.warn("bright field image not be provided, the result may not accurate")
    raise ValueError("bf image must be given!")
else:
    bf = args.bf
if args.o is False:
    output = os.path.basename(pcna.replace('.tif', '.json'))
    print(f"-o  not provided, using the default output file name: {output}")
else:
    print(type(args.o))
    if not args.o.endswith('.json'):
        raise ValueError("output filename need <.json> extend name")
    output = args.o
#if config.TIMES == 20:
#    segment_model = None
#elif config.TIMES == 60:
#    segment_model = StarDist2D(None, name='stardist_no_shape_completion_60x_seg',
#                   basedir='/home/zje/CellClassify/ResNet-Tensorflow/models_60x_seg')
#else:
#    raise ValueError(f"Image magnification should be 20 or 60, got {config.TIMES} instead")
prediction.segment(pcna=pcna, bf=bf, output=output, segment_model=None)
