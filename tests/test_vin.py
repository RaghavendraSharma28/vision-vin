from mulkiya_vin.vin import clean_chassis_no, is_valid_chassis_format, validate_chassis_no


def test_valid_17_character_chassis_number():
    result = validate_chassis_no("MNTBAAB17L0001234")

    assert result["cleaned"] == "MNTBAAB17L0001234"
    assert result["format_valid"] is True


def test_too_short_chassis_number_is_invalid():
    result = validate_chassis_no("MNT123")

    assert result["length_valid"] is False
    assert result["format_valid"] is False


def test_forbidden_i_o_q_characters_are_invalid():
    assert is_valid_chassis_format("MNTBAABI7L0001234") is False
    assert is_valid_chassis_format("MNTBAABO7L0001234") is False
    assert is_valid_chassis_format("MNTBAABQ7L0001234") is False


def test_lowercase_input_gets_cleaned():
    assert clean_chassis_no("mntbaab17l0001234") == "MNTBAAB17L0001234"


def test_spaces_and_hyphens_are_removed():
    assert clean_chassis_no("MNT BAA-B17 L00-01234") == "MNTBAAB17L0001234"
