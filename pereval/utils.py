import json
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Union


def calculate_ci(values):
    ci = 0.0
    std_dev = np.std(values, ddof=1)
    n = len(values)
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_value = stats.t.ppf(1 - alpha/2, df=n-1)
    std_err = std_dev / np.sqrt(n)
    ci = t_value * std_err
    return ci


def load_json(json_path: str) -> Union[Dict, List[Dict]]:
    """Loads json.

    Args:
        json_path: The path to json.

    Returns:
        The content of json file.
    """
    with open(json_path, "r", encoding="utf8") as f:
        json_content = json.load(f)
    return json_content
