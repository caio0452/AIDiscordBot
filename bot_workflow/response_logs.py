class ResponseLogsManager:
    def __init__(self, log_capacity: int = 10):
        self.log_capacity = log_capacity
        self._last_message_id_logs: dict[int, str] = {}

    def store_log(self, message_id: int, log: str):
        print(f"Saved log for message id {message_id}")
        self._last_message_id_logs[message_id] = log
        if len(self._last_message_id_logs) > self.log_capacity:
            oldest_key = next(iter(self._last_message_id_logs))
            del self._last_message_id_logs[oldest_key]

    def get_log_by_id(self, message_id: int) -> str | None:
        return self._last_message_id_logs.get(message_id, None)