"""Example prompt tests.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    uv run pytest-prompts run examples/
"""
from pytest_prompts import prompt_test


@prompt_test()
def test_summary_is_concise(runner):
    result = runner.run(
        prompt="examples/prompts/summarizer.txt",
        input=(
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
            "in Paris, France. It was constructed from 1887 to 1889 as the centerpiece "
            "of the 1889 World's Fair. It stands 330 metres tall."
        ),
    )
    assert len(result.output.split()) < 80
    assert result.latency_ms < 10_000


@prompt_test()
def test_qa_knows_capital_of_france(runner):
    result = runner.run(
        prompt="examples/prompts/qa.txt",
        input="What is the capital of France?",
    )
    assert "Paris" in result.output


@prompt_test()
def test_qa_admits_uncertainty(runner):
    result = runner.run(
        prompt="examples/prompts/qa.txt",
        input="What did I eat for breakfast on 2019-04-02?",
    )
    assert "don't know" in result.output.lower() or "cannot" in result.output.lower()
