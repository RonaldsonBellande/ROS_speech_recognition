from header_imports import *

if __name__ == "__main__":
    
    if len(sys.argv) != 1:

        if sys.argv[1] == "model_building":
            speech__analysis_obj = model_building(model_type=sys.argv[2], data_type=sys.argv[3])

        if sys.argv[1] == "model_training":
            speech_analysis_obj = model_training(model_type=sys.argv[2], data_type=sys.argv[3])
        
        if sys.argv[1] == "transfer_learning":
            speech_analysis_obj = transfer_learning(model_type=sys.argv[2], data_type=sys.argv[3])

        if sys.argv[1] == "continuous_learning":
            speech_analysis_obj = continuous_learning(model_type=sys.argv[2], data_type=sys.argv[3])



