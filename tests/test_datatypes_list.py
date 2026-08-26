from nematics3d.datatypes import ensure_list


def test_list_is_returned_unchanged():
    value = [1, 2]
    assert ensure_list(value) is value


def test_tuple_and_set_are_expanded():
    assert ensure_list((1, 2)) == [1, 2]
    assert set(ensure_list({1, 2})) == {1, 2}


def test_other_objects_are_treated_as_single_items():
    text = "abc"
    sequence = range(3)
    generator = (value for value in range(3))

    assert ensure_list(text) == [text]
    assert ensure_list(sequence) == [sequence]
    assert ensure_list(generator) == [generator]


def test_none_is_a_valid_single_item():
    assert ensure_list(None) == [None]
