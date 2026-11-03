"""Uses only the dedicated integration containers configured by compose.tests.yml."""
import os
import uuid

import psycopg
import pytest
import redis
from src.memory.store import ConversationMemoryStore

pytestmark = pytest.mark.skipif(not os.getenv('OMNIRAG_TEST_POSTGRES_DSN'), reason='Isolated integration database not configured')


def test_archive_restores_summary_after_hot_session_ends():
    dsn = os.environ['OMNIRAG_TEST_POSTGRES_DSN']
    hot = redis.Redis.from_url(os.environ['OMNIRAG_TEST_REDIS_URL'], decode_responses=True)
    store = ConversationMemoryStore(redis_client=hot, connection_factory=lambda: psycopg.connect(dsn))
    owner, session = 'test-' + uuid.uuid4().hex, uuid.uuid4().hex
    assert store.save_summary(owner, session, 'synthetic project context') == 'redis'
    assert store.load_summary(owner, session).source == 'redis'
    assert store.end_session(session)
    restored = store.load_summary(owner, 'restored-' + session)
    assert restored.source == 'postgres' and restored.summary == 'synthetic project context'
    assert store.load_summary('another-' + owner, 'restored-' + session) is None
