import json


def get_config(tag: str):
    key_list = tag.split(".")
    cur = _config
    for key_item in key_list:
        if key_item == "":
            continue
        if key_item in cur:
            cur = cur[key_item]
        else:
            raise ValueError("Not exist config key: " + tag)
    return cur


with open("./config.json", "r", encoding="utf8") as _f_cfg:
    _config = json.load(_f_cfg)
