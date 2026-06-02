"""
进度推送工具 —— OpenDetect_AI
各 Agent 节点调用 push_progress() 写入进度消息，
api.py 的 SSE 生成器通过 get_progress_queue() 消费。
"""
import queue
import threading

# 每个请求一个独立队列，用 thread_id 隔离
_queues: dict[str, queue.Queue] = {}
_lock = threading.Lock()


def get_or_create_queue(thread_id: str) -> queue.Queue:
    with _lock:
        if thread_id not in _queues:
            _queues[thread_id] = queue.Queue()
        return _queues[thread_id]


def push_progress(thread_id: str, message: str) -> None:
    """Agent 节点调用：推送一条进度消息。"""
    q = get_or_create_queue(thread_id)
    q.put_nowait(message)


def drain_queue(thread_id: str) -> list[str]:
    """api.py 调用：取出当前队列里所有待推送的消息（非阻塞）。"""
    q = get_or_create_queue(thread_id)
    messages = []
    while True:
        try:
            messages.append(q.get_nowait())
        except queue.Empty:
            break
    return messages


def cleanup_queue(thread_id: str) -> None:
    """请求结束后清理队列。"""
    with _lock:
        _queues.pop(thread_id, None)