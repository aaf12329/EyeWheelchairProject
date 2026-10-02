"""CSV 记录器：对比模式把双引擎逐帧数据落盘（数据文件，不入库）。

定位：I/O 单一入口原则里的"结构化数据口"——双引擎对比脚本共用这一个出口，
不各自手写 open/csv.writer（改路径、改格式只改这一处）。
"""
import csv
import time
from pathlib import Path


class CsvLogger:
    """打开一个带时间戳的 CSV，逐帧写行；用完 close()。

    用法：
        log = CsvLogger(COMPARE_LOG_DIR, "blink_compare", ["t_s", "frame", ...])
        log.log([0.03, 1, ...])
        log.close()
    """

    def __init__(self, directory: Path, prefix: str, header: list[str]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = directory / f"{prefix}_{stamp}.csv"
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)
        self._rows = 0

    def log(self, row: list) -> None:
        """写一行（None 会被写成空串，方便 Excel/pandas 读）。"""
        self._writer.writerow(["" if v is None else v for v in row])
        self._rows += 1

    def close(self) -> None:
        self._file.close()

    @property
    def rows(self) -> int:
        return self._rows
