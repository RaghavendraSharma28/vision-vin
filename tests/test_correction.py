from mulkiya_vin.correction import score_candidate_against_ground_truth_pool


def test_repairs_missing_first_character():
    known = ["JN8AT3MT5PW009001"]

    result = score_candidate_against_ground_truth_pool("N8AT3MT5PW009001", known)

    assert result is not None
    assert result["matched_chassis_no"] == "JN8AT3MT5PW009001"


def test_repairs_small_ocr_style_mistake():
    known = ["3PCAJ5M3XRF103525"]

    result = score_candidate_against_ground_truth_pool("3PCAJS5M3XRF103525", known)

    assert result is not None
    assert result["matched_chassis_no"] == "3PCAJ5M3XRF103525"


def test_repairs_near_length_candidate():
    known = ["JN6BE6CS9T9440421"]

    result = score_candidate_against_ground_truth_pool("JN6BE6CS09440421", known)

    assert result is not None
    assert result["matched_chassis_no"] == "JN6BE6CS9T9440421"


def test_short_unrelated_candidate_does_not_repair():
    known = ["VF1HSRSC5KA524121"]

    result = score_candidate_against_ground_truth_pool("C192771", known)

    assert result is None


def test_random_short_candidate_does_not_repair():
    known = ["5N1DR3MR9RC244870"]

    result = score_candidate_against_ground_truth_pool("VQ35815756W", known)

    assert result is None
