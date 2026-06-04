from __future__ import annotations

from dataclasses import dataclass

from binary_mopso_cd.services.turbulence import TurbulenceService


@dataclass
class StubToken:
    text: str
    lemma_: str
    pos_: str
    idx: int
    is_stop: bool = False
    is_punct: bool = False
    is_space: bool = False
    like_num: bool = False


class StubNLP:
    def __init__(self, tokens: list[StubToken]):
        self.tokens = tokens

    def __call__(self, _text: str) -> list[StubToken]:
        return self.tokens


def make_service(tokens: list[StubToken]) -> TurbulenceService:
    service = TurbulenceService(distilbert=None, wordnet=None, spacy_model="stub")
    service._nlp = StubNLP(tokens)
    return service


def test_turbulence_units_prefer_component_pos_when_available():
    service = make_service(
        [
            StubToken("local", "local", "ADJ", 0),
            StubToken("officials", "official", "NOUN", 6),
            StubToken("warn", "warn", "VERB", 16),
            StubToken("residents", "resident", "NOUN", 21),
            StubToken("quickly", "quickly", "ADV", 31),
        ]
    )

    units = service.modifiable_units("local officials warn residents quickly", ("NOUN", "PROPN"))

    assert [unit["text"] for unit in units] == ["officials", "residents"]
    assert all(unit["pos"] == "NOUN" for unit in units)
    assert units[0]["target_index"] == 1
    assert units[0]["span"] == [6, 15]


def test_turbulence_units_fall_back_to_all_valid_pos_when_no_preferred_units():
    service = make_service(
        [
            StubToken("local", "local", "ADJ", 0),
            StubToken("warn", "warn", "VERB", 6),
            StubToken("quickly", "quickly", "ADV", 11),
        ]
    )

    units = service.modifiable_units("local warn quickly", ("NOUN", "PROPN"))

    assert [unit["pos"] for unit in units] == ["ADJ", "VERB", "ADV"]


def test_turbulence_units_skip_stopwords_numbers_spaces_and_punctuation():
    service = make_service(
        [
            StubToken("the", "the", "DET", 0, is_stop=True),
            StubToken("12", "12", "NUM", 4, like_num=True),
            StubToken(",", ",", "PUNCT", 6, is_punct=True),
            StubToken("alert", "alert", "NOUN", 8),
        ]
    )

    units = service.modifiable_units("the 12, alert", ("NOUN",))

    assert [unit["text"] for unit in units] == ["alert"]
