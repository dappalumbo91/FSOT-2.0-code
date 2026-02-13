import json
from mast_api import build_mast_request, mast_cone_search


def test_build_mast_request_structure():
    r = build_mast_request("Mast.Caom.Cone", {"ra":1.0, "dec":2.0, "radius":0.1}, fmt="json", pagesize=10)
    assert isinstance(r, dict)
    assert r["service"] == "Mast.Caom.Cone"
    assert r["params"]["ra"] == 1.0
    assert r["format"] == "json"


def test_mast_cone_search_builder():
    obj = mast_cone_search(10.0, -10.0, 0.05, pagesize=5)
    assert obj["service"] == "Mast.Caom.Cone"
    assert obj["params"]["radius"] == 0.05
    assert obj["pagesize"] == 5
