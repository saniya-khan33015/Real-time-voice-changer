import logging


class _Logger:
    def __init__(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self._logger = logging.getLogger("local_voice_studio")
        self._logger.propagate = False
        self._handlers = []

    def _format(self, message, *args):
        if args:
            try:
                return str(message).format(*args)
            except Exception:
                return f"{message} {' '.join(map(str, args))}"
        return message

    def info(self, message, *args, **kwargs):
        self._logger.info(self._format(message, *args))

    def warning(self, message, *args, **kwargs):
        self._logger.warning(self._format(message, *args))

    def error(self, message, *args, **kwargs):
        self._logger.error(self._format(message, *args))

    def debug(self, message, *args, **kwargs):
        self._logger.debug(self._format(message, *args))

    def exception(self, message, *args, **kwargs):
        self._logger.exception(self._format(message, *args))

    def remove(self, *args, **kwargs):
        for handler in self._handlers:
            self._logger.removeHandler(handler)
        self._handlers = []

    def add(self, sink, level="INFO", **kwargs):
        if hasattr(sink, "write"):
            handler = logging.StreamHandler(sink)
        else:
            handler = logging.FileHandler(sink, encoding="utf-8")
        handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        self._handlers.append(handler)
        return len(self._handlers) - 1


logger = _Logger()
