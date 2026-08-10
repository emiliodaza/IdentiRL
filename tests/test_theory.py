from identirl.envs import Condition
from identirl.theory import factorial_value_gaps, matched_value_gaps, validate_factorial_labels


def test_four_factorial_labels_are_realized():
    assert validate_factorial_labels() == {
        "neither": (False, False),
        "reward": (True, False),
        "observation": (False, True),
        "both": (True, True),
    }


def test_value_gaps_have_expected_closed_forms():
    assert factorial_value_gaps(Condition.NEITHER).as_dict() == {
        "reward_gap": 0.0,
        "observation_gap": 0.0,
        "memory_gap": 0.0,
    }
    assert factorial_value_gaps(Condition.BOTH).reward_gap == 2.5
    assert factorial_value_gaps(Condition.BOTH).observation_gap == 1.25
    assert matched_value_gaps("observation").observation_gap == 1.25
    assert matched_value_gaps("reward").reward_gap == 5.0
