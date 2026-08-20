from src.data.split import build_reproduction, build_strict


def rows():
    result = []
    for bearing, label in (("K001", "0"), ("KA04", "1"), ("KI04", "2"), ("K002", "0"), ("KA15", "1"), ("KI14", "2"), ("K003", "0"), ("KA16", "1"), ("KI16", "2")):
        for index in range(10):
            result.append({"sample_id": f"{bearing}_{index}", "bearing_id": bearing, "label": label, "source_file": f"{bearing}_{index}.mat"})
    return result


def test_strict_splits_have_no_bearing_leakage():
    folds = build_strict(rows(), "D:/code/putest/pu_experiment/.test_outputs")
    assert len(folds) == 3
    for fold in folds:
        sets = [set(fold["bearings"][name]) for name in ("train", "val", "test")]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_reproduction_covers_all_samples():
    payload = build_reproduction(rows(), "D:/code/putest/pu_experiment/.test_outputs/repro.json")
    ids = sum(payload["partitions"].values(), [])
    assert len(ids) == len(rows())
    assert len(set(ids)) == len(rows())
