from ifls.normalize import normalize


def test_lowercase():
    assert normalize("POULET RÔTI") == "poulet roti"


def test_strip_accents():
    assert normalize("émincé") == "emince"


def test_strip_weight():
    assert normalize("jambon 500g") == "jambon"
    assert normalize("lait 1 l") == "lait"
    assert normalize("beurre 250 g") == "beurre"


def test_strip_compound_weight():
    assert normalize("steak 2/5 kg") == "steak"
    assert normalize("steak 2.5 kg") == "steak"


def test_pack_normalization():
    assert normalize("yaourt x 6") == "yaourt bt"
    assert normalize("biscuits x6") == "biscuits bt"


def test_collapse_spaces():
    assert normalize("poulet   roti") == "poulet roti"


def test_empty():
    assert normalize("") == ""
