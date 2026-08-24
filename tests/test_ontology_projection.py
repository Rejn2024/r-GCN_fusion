from rgcn_fusion.ontology_projection import ESM, GEO, PROV, materialize_ontology_view


class _Result:
    def consume(self):
        return self

    def single(self):
        return {"resources": 17}


class _Session:
    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, query, **parameters):
        self.calls.append((query, parameters))
        return _Result()


class _Driver:
    def __init__(self):
        self.opened = None

    def session(self, **kwargs):
        self.opened = _Session()
        return self.opened


def test_projection_materialises_all_three_standard_vocabularies():
    driver = _Driver()
    assert materialize_ontology_view(driver, "neo4j") == {"ontology_resources": 17}
    source = "\n".join(query for query, _ in driver.opened.calls)
    assert "ESMObservation:ProvEntity" in source
    assert "PROV_WAS_DERIVED_FROM" in source
    assert "Geometry" in source and "CRS84" in source
    for _, parameters in driver.opened.calls[:-1]:
        assert parameters["esm"] == ESM
        assert parameters["prov"] == PROV
        assert parameters["geo"] == GEO


def test_population_notebooks_apply_ontology_projection():
    from pathlib import Path

    for name in (
        "neo4j_kg_creation.ipynb",
        "observation_etl_rgcn_end_to_end.ipynb",
        "neo4j_observation_series_visualisation.ipynb",
    ):
        source = (Path("notebooks") / name).read_text(encoding="utf-8")
        assert "materialize_ontology_view" in source
