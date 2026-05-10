"""Tests for the natural-language memory search agent."""

from aeon_v1 import Config, ingest
from aeon_v1.search_agent import SearchAgent


def test_search_agent_detects_memory_search_intent(tmp_path):
    cfg = Config(tmp_path)
    agent = SearchAgent(cfg)

    assert agent.is_search_query("what do you remember about Aeon")
    assert not agent.is_search_query("good morning")


def test_search_agent_answers_from_local_memory(tmp_path):
    cfg = Config(tmp_path)
    ingest("Aeon is a local-first AI memory project with Obsidian integration.", config=cfg)
    agent = SearchAgent(cfg)

    reply = agent.handle_chat_query("what do you remember about Aeon")

    assert reply is not None
    assert "local-first AI memory project" in reply


def test_search_agent_self_query_does_not_get_stuck_on_test_fact(tmp_path):
    cfg = Config(tmp_path)
    ingest("The random number the operator asked Aeon to remember as an important test memory is 123456.", config=cfg)
    ingest(
        "Aeon is a local recursive dashboard memory project with Obsidian and LM Studio.",
        config=cfg,
    )
    agent = SearchAgent(cfg)

    reply = agent.handle_chat_query("what do you remember about Aeon")

    assert reply is not None
    assert "local recursive dashboard memory project" in reply


def test_search_agent_programmatic_results_preserve_raw_records(tmp_path):
    cfg = Config(tmp_path)
    ingest("The special dashboard recall number is 123456.", config=cfg)
    agent = SearchAgent(cfg)

    results = agent.results("123456", limit=5)

    assert any(r.get("memory", {}).get("type") == "raw" for r in results)


def test_search_agent_filters_prior_search_questions(tmp_path):
    cfg = Config(tmp_path)
    ingest("User: what do you remember about CANARY-20260509-AXLE-BLUE-731", config=cfg)
    ingest(
        "CANARY-20260509-AXLE-BLUE-731: the operator asked Aeon to remember riverglass screwdriver.",
        config=cfg,
    )
    agent = SearchAgent(cfg)

    reply = agent.handle_chat_query("what do you remember about CANARY-20260509-AXLE-BLUE-731")

    assert reply is not None
    assert "riverglass screwdriver" in reply
    assert "User: what do you remember" not in reply


def test_search_agent_answers_number_recall_from_raw_memory(tmp_path):
    cfg = Config(tmp_path)
    ingest("User: ok wait, that number 123456 was a stand alone remember just as a test", config=cfg)
    ingest("Aeon: I do not remember a number. I only remember context.", config=cfg)
    agent = SearchAgent(cfg)

    result = agent.handle_chat_query_with_ids("what number are you supposed to remember?")

    assert result is not None
    assert result["reply"].startswith("The number is 123456.")
    assert "only remember context" not in result["reply"]


def test_search_agent_answers_six_digit_number_request(tmp_path):
    cfg = Config(tmp_path)
    ingest("Aeon: The number you asked me to remember is 123456.", config=cfg)
    agent = SearchAgent(cfg)

    result = agent.handle_chat_query_with_ids("search the raw inputs for a 6 digit number")

    assert result is not None
    assert result["reply"].startswith("The number is 123456.")


def test_search_agent_answers_phrase_recall_with_timestamp(tmp_path):
    cfg = Config(tmp_path)
    ingest(
        'User: remember the phrase "Have a little cup of Liber-TEA!" '
        "and return the phrase plus the date and time I gave it to you.",
        config=cfg,
    )
    agent = SearchAgent(cfg)

    result = agent.handle_chat_query_with_ids("search for phrase i asked you to remember")

    assert result is not None
    assert 'The phrase is "Have a little cup of Liber-TEA!"' in result["reply"]
    assert "Received at:" in result["reply"]
    assert "Source: [raw]" in result["reply"]


def test_search_agent_direct_phrase_question_does_not_use_number_path(tmp_path):
    cfg = Config(tmp_path)
    ingest('User: remember the phrase "riverglass screwdriver" for a test.', config=cfg)
    agent = SearchAgent(cfg)

    result = agent.handle_chat_query_with_ids("what phrase are you supposed to remember?")

    assert result is not None
    assert 'The phrase is "riverglass screwdriver"' in result["reply"]


def test_search_agent_uses_llm_planned_queries(tmp_path, monkeypatch):
    cfg = Config(tmp_path)
    ingest("An online critic said Aeon is just a memory.txt file, which is not accurate.", config=cfg)
    agent = SearchAgent(cfg)

    monkeypatch.setattr(
        "aeon_v1.search_agent.generate_search_text",
        lambda prompt, config: '{"queries":["memory.txt file online critic"]}',
    )

    results = agent.results("refute the comment", limit=5)

    assert any("memory.txt file" in r.get("memory", {}).get("text", "") for r in results)
