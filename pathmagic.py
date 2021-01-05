import os
import sys

PROJECT_DIR = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_DIR, 'data'))
sys.path.append(os.path.join(PROJECT_DIR, 'loss'))
sys.path.append(os.path.join(PROJECT_DIR, 'models'))
sys.path.append(os.path.join(PROJECT_DIR, 'models/expansion_penalty'))
sys.path.append(os.path.join(PROJECT_DIR, 'models/MDS'))
sys.path.append(os.path.join(PROJECT_DIR, 'models/emd'))
