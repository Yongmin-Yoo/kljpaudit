from kljpaudit.labels import (
    all_index_to_raw,
    validate_expected_labels,
)


def test_mapping_first_label_structure():
    data = {
        "raw_label_values": {
            "fine_lv": [0, 1, 2, 3, 4],
            "imprisonment_with_labor_lv": [0, 1, 2, 3, 4, 5],
            "imprisonment_without_labor_lv": [0, 1, 2, 3, 4],
        },
        "index_to_label": {
            "fine_lv": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
            "imprisonment_with_labor_lv": {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "imprisonment_without_labor_lv": {
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
            },
        },
    }

    mappings = all_index_to_raw(data)
    validate_expected_labels(mappings)

    assert mappings["fine_lv"][4] == 4
    assert mappings["imprisonment_with_labor_lv"][5] == 5
