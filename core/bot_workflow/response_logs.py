import datetime

class ResponseLogger:
    def __init__(self):
        self.text = ""

    def verbose(self, text: str, *, category: str | None = None):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if category:
            self.text += f"\n[{current_time}] --- {category} ---\n{text}\n"
        else:
            self.text += f"\n[{current_time}] {text}\n"

class ResponseLogsManager:
    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            print('Creating new instance')
            cls._instance = cls.__new__(cls)
            cls.log_capacity = 10
            cls._last_message_id_logs: dict[int, str] = {}
        return cls._instance
    
    def store_log(self, message_id: int, log: str):
        print(f"Saved log for message id {message_id}")
        self._last_message_id_logs[message_id] = log
        if len(self._last_message_id_logs) > self.log_capacity:
            oldest_key = next(iter(self._last_message_id_logs))
            del self._last_message_id_logs[oldest_key]

    def get_log_by_id(self, message_id: int) -> str | None:
        return self._last_message_id_logs.get(message_id, None)
