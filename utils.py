import soundfile as sf
import rapidjson as rj

def bufspill(audio_file_path: str):
    """Returns an audio files fp32 values as a flat numpy array"""
    data, sr = sf.read(audio_file_path)
    data = data.transpose()
    return data

def write_json(json_file_path: str, in_dict: dict):
    """Takes a dictionary and writes it to JSON file"""
    with open(json_file_path, "w+") as fp:
        rj.dump(in_dict, fp, indent=4)


def read_json(json_file_path: str) -> dict:
    """Takes a JSON file and returns a dictionary"""
    with open(json_file_path, "r") as fp:
        data = rj.load(fp)
        return data

    
