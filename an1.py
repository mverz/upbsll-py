import eos
import sys

ANALYSIS_FILE='./an1.yaml'
af=eos.AnalysisFile(ANALYSIS_FILE)

# PROMPR USER FOR WHICH DECAY TO SAMPLE
if 1:
    print("- 0 to continue without sampling\n- 1 for B to K ll\n- 2 for B to K* ll\n- 3 for Bs to Phi ll\n- 99 for all decays")
    decay_select = input()

    match decay_select:
        case "0":
            print("Continuing without sampling.")
        case "1":
            eos.tasks.sample_nested(af,'BSZ-BqToK-wSR', base_directory='./data_base/', nlive=500, dlogz=0.5, seed=42)
        case "2":
            eos.tasks.sample_nested(af,'BSZ-BqToK*-wSR', base_directory='./data_base/', nlive=500, dlogz=0.5, seed=42)
        case "3":
            eos.tasks.sample_nested(af,'BSZ-BsToPhi-wSR', base_directory='./data_base/', nlive=500, dlogz=0.5, seed=42)
        case "99":
            eos.tasks.sample_nested(af,'BSZ-BqToK-wSR', base_directory='./data_base/', nlive=500, dlogz=0.5, seed=42)
            eos.tasks.sample_nested(af,'BSZ-BqToK*-wSR', base_directory='./data_base/', nlive=500, dlogz=0.5, seed=42)
            eos.tasks.sample_nested(af,'BSZ-BsToPhi-wSR', base_directory='./data_base/', nlive=500, dlogz=0.5, seed=42)
        case _:
            print("No valid input. Exiting.")
            sys.exit()


# EXTRACT OBSERVABLE PREDICTIONS
if 1:
    print("- 0 to continue without predictions\n- 1 for B to K ll\n- 2 for B to K* ll\n- 3 for Bs to Phi ll\n- 99 for all decays")
    decay_select = input()

    match decay_select:
        case "0":
            print("Continuing without predictions.")
        case "1":
            predicts=['BToK-f+', 'BToK-f0', 'BToK-fT']
            for i in predicts:
                eos.tasks.predict_observables(af, 'BSZ-BqToK-wSR', i,  base_directory='./data_base/')    
        case "2":
            predicts=['BToK*-A0', 'BToK*-A1', 'BToK*-A12', 'BToK*-V', 'BToK*-T1', 'BToK*-T2', 'BToK*-T23']
            for i in predicts:
                eos.tasks.predict_observables(af, 'BSZ-BqToK*-wSR', i,  base_directory='./data_base/')    
        case "3":
            predicts=['BsToPhi-A0', 'BsToPhi-A1', 'BsToPhi-A12', 'BsToPhi-V', 'BsToPhi-T1', 'BsToPhi-T2', 'BsToPhi-T23']
            for i in predicts:
                eos.tasks.predict_observables(af, 'BSZ-BsToPhi-wSR', i,  base_directory='./data_base/')
        case "99":
            predicts=['BToK-f+', 'BToK-f0', 'BToK-fT']
            for i in predicts:
                eos.tasks.predict_observables(af, 'BSZ-BqToK-wSR', i,  base_directory='./data_base/')
            predicts=['BToK*-A0', 'BToK*-A1', 'BToK*-A12', 'BToK*-V', 'BToK*-T1', 'BToK*-T2', 'BToK*-T23']
            for i in predicts:
                eos.tasks.predict_observables(af, 'BSZ-BqToK*-wSR', i,  base_directory='./data_base/')    

            predicts=['BsToPhi-A0', 'BsToPhi-A1', 'BsToPhi-A12', 'BsToPhi-V', 'BsToPhi-T1', 'BsToPhi-T2', 'BsToPhi-T23']
            for i in predicts:
                eos.tasks.predict_observables(af, 'BSZ-BsToPhi-wSR', i,  base_directory='./data_base/')
        case _:
            print("No valid input. Exiting.")
            sys.exit()
