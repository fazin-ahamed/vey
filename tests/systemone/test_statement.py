import pytest

from vey.systemone.nli import to_statement


@pytest.mark.parametrize("question,statement", [
    ("Is this message about billing?", "This message is about billing."),
    ("Does this message express urgency?", "This message expresses urgency."),
    ("Is the sentiment of this movie review positive?", "The sentiment of this movie review is positive."),
    ("Is the customer asking for a refund?", "The customer is asking for a refund."),
    ("Can the customer log in?", "The customer can log in."),
    ("Was the package delivered on time?", "The package was delivered on time."),
    ("Has the issue been resolved?", "The issue has been resolved."),
    ("Is this a billing issue?", "This is a billing issue."),
    ("Do these reviews mention shipping?", "These reviews mention shipping."),
    ("Did the customer pay the invoice?", "The customer paid the invoice."),
    ("Does the user threaten to cancel or leave?", "The user threatens to cancel or leave."),
])
def test_questions_become_statements(question, statement):
    assert to_statement(question) == statement


@pytest.mark.parametrize("text", [
    "This message contains unsolicited advertising.",  # already a statement
    "Which team should handle this?",                    # not a yes/no question
    "Is it?",                                            # too short to rewrite safely
    "",
])
def test_everything_else_is_unchanged(text):
    assert to_statement(text) == text
