"""Tests for JSON repair in parse_arguments (small-model tool call resilience)."""


from agent_mini.providers.base import parse_arguments


class TestParseArgumentsRepair:
    def test_valid_json(self):
        assert parse_arguments('{"path": "test.txt"}') == {"path": "test.txt"}

    def test_trailing_comma_object(self):
        assert parse_arguments('{"path": "test.txt",}') == {"path": "test.txt"}

    def test_trailing_comma_array_value(self):
        result = parse_arguments('{"items": [1, 2, 3,]}')
        assert result == {"items": [1, 2, 3]}

    def test_single_quotes(self):
        assert parse_arguments("{'path': 'test.txt'}") == {"path": "test.txt"}

    def test_unquoted_keys(self):
        assert parse_arguments('{path: "test.txt"}') == {"path": "test.txt"}

    def test_markdown_code_fence(self):
        raw = '```json\n{"path": "test.txt"}\n```'
        assert parse_arguments(raw) == {"path": "test.txt"}

    def test_dict_passthrough(self):
        assert parse_arguments({"key": "val"}) == {"key": "val"}

    def test_empty_string(self):
        assert parse_arguments("") == {}

    def test_completely_broken(self):
        assert parse_arguments("not json at all {{{") == {}

    def test_non_dict_json(self):
        # A JSON array should not be accepted
        assert parse_arguments("[1, 2, 3]") == {}
