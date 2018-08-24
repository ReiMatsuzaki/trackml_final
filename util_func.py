import os

def get_path_to_submission(path_to_dir, event_id):
    if(path_to_dir=="none"):
        return None
    else:
        return os.path.join(path_to_dir, "event{0:0>10}-submission.csv".format(event_id))
