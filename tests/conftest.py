import pytest

from ledgerlens.data import build_ledger


@pytest.fixture(scope="session")
def con():
    return build_ledger()
