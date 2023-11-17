import copy
import json
import sys

if __name__ == "__main__":
    if len(sys.argv) == 4:
        rs_file = sys.argv[1]
        xamr_file = sys.argv[2]
        out_file = sys.argv[3]

        rs_json = json.load(open(rs_file))
        xamr_json = json.load(open(xamr_file))

        m_id2rs = {t["mention_id"]: t for t in rs_json}
        m_id2xamr = {t["mention_id"]: t for t in xamr_json}

        mids = list(m_id2xamr.keys())

        my_tasks = []
        arg_names = ["arg0", "arg1", "argL", "argT"]
        for m_id in mids:
            t1 = m_id2xamr[m_id]
            t2 = m_id2rs[m_id]
            if t1["mention_id"] != t2["mention_id"]:
                raise AssertionError
            for arg_name in arg_names:
                if "predicted" not in t2:
                    t2["predicted"] = {"roleset_id": ""}
                t2["predicted"][arg_name] = t1[arg_name + "_pred"]
                t2[arg_name] = t1[arg_name]
            t1["predicted"] = t2["predicted"]
            my_tasks.append(t1)
        json.dump(my_tasks, open(out_file, "w"), indent=1)

    else:
        print("arguments size should be 4")
        exit(0)
