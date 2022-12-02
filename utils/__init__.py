import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or any {‘0’, ‘1’, ‘2’}, stops tf from complaining about things it shouldn't be worried about

from utils.conf import *
from utils.eval import Eval, isSyntacticallyCorrect
from utils.StudentTeacher import Distiller, FnLayer, mystery_function, construct_student_big, FnsLayer, construct_student_AST, haltCallback
from utils.nn import getNN, iterateNN, plotNN, bf_unit_nns, loadNNs
