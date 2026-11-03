import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from src.graph.traversal import QdrantGraphTraverser
from src.memory.store import ConversationMemoryStore
from src.memory.summary import DialogueSummarizer


@pytest.fixture
def graph_client():
    client = QdrantClient(':memory:')
    client.create_collection('demo', vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    client.upsert('demo', points=[
        models.PointStruct(id=1, vector=[1.0, 0.0], payload={'page_content': 'A', 'metadata': {'chunk_id': 'a', 'related_ids': ['b']}}),
        models.PointStruct(id=2, vector=[1.0, 0.0], payload={'page_content': 'B', 'metadata': {'chunk_id': 'b', 'related_ids': ['a', 'c']}}),
        models.PointStruct(id=3, vector=[1.0, 0.0], payload={'page_content': 'C', 'metadata': {'chunk_id': 'c', 'related_ids': ['a']}}),
    ])
    yield client
    client.close()


@pytest.mark.parametrize('depth,expected', [(0, ['A']), (1, ['A', 'B']), (2, ['A', 'B', 'C'])])
def test_bfs_respects_depth_and_does_not_repeat_cycles(graph_client, depth, expected):
    seed = Document(page_content='A', metadata={'chunk_id': 'a'})
    result = QdrantGraphTraverser(graph_client, 'demo', max_depth=depth).expand([seed])
    assert [doc.page_content for doc in result] == expected


def test_bfs_respects_extra_node_limit(graph_client):
    seed = Document(page_content='A', metadata={'chunk_id': 'a'})
    result = QdrantGraphTraverser(graph_client, 'demo', max_depth=10, max_nodes=1).expand([seed])
    assert [doc.page_content for doc in result] == ['A', 'B']


def test_graph_failure_preserves_original_context():
    client = MagicMock()
    client.scroll.side_effect = RuntimeError('test-only unavailable store')
    seeds = [Document(page_content='A', metadata={'chunk_id': 'a'})]
    assert QdrantGraphTraverser(client, 'demo').expand(seeds) == seeds


def memory_store(redis_client=None, connection=None):
    return ConversationMemoryStore(redis_client=redis_client or MagicMock(), connection_factory=connection or MagicMock())


def test_hot_memory_rejects_another_user():
    redis = MagicMock()
    redis.get.return_value = json.dumps({'user_id': 'other', 'summary': 'private test context'})
    store = memory_store(redis_client=redis)
    assert store._load_hot('owner', 'session') is None
    redis.expire.assert_not_called()


def test_hot_memory_refreshes_ttl_for_correct_user():
    redis = MagicMock()
    redis.get.return_value = json.dumps({'user_id': 'owner', 'summary': 'context', 'updated_at': 'now'})
    store = memory_store(redis_client=redis)
    assert store._load_hot('owner', 'session').summary == 'context'
    redis.expire.assert_called_once_with('omni-rag:memory:session', store.ttl_seconds)


def test_cold_memory_is_restored_into_new_session():
    redis = MagicMock()
    redis.get.return_value = None
    connect = MagicMock()
    cursor = connect.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = ('old-session', 'archived context', datetime.now(timezone.utc))
    store = memory_store(redis, connect)
    result = store.load_summary('owner', 'new-session')
    assert result.session_id == 'new-session' and result.source == 'postgres'
    payload = json.loads(redis.setex.call_args.args[2])
    assert payload['summary'] == 'archived context'
    assert payload['user_id'] == 'owner'


def test_archive_failure_does_not_prevent_hot_write():
    redis = MagicMock()
    connect = MagicMock(side_effect=RuntimeError('isolated test failure'))
    store = memory_store(redis, connect)
    assert store.save_summary('owner', 'session', 'context') == 'redis'
    redis.setex.assert_called_once()


def test_ending_session_keeps_archive():
    connect = MagicMock()
    store = memory_store(connection=connect)
    assert store.end_session('session')
    store.redis.delete.assert_called_once_with('omni-rag:memory:session')
    connect.assert_not_called()


def test_empty_summary_keeps_previous_context():
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content='  ')
    assert DialogueSummarizer(llm).summarize('previous', 'question', 'answer') == 'previous'


def test_summary_is_bounded_and_contains_new_turn_in_prompt():
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content='x' * 100)
    assert len(DialogueSummarizer(llm, max_summary_chars=20).summarize('previous', 'question', 'answer')) == 20
    assert 'question' in llm.invoke.call_args.args[0]
