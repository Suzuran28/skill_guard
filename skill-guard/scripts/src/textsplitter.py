from __future__ import annotations

import re
from typing import Callable, Iterable


class TextSplitter:
    """文本分割器基类。

    提供将长文本按指定大小切分为 chunk 的基础能力，子类需实现 split_text 方法。
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = True,
    ) -> None:
        """初始化文本分割器。

        Args:
            chunk_size (int): 每个 chunk 的最大长度，默认 1000。
            chunk_overlap (int): 相邻 chunk 之间的重叠长度，默认 200，须小于 chunk_size。
            length_function (Callable[[str], int]): 计算文本长度的函数，默认为 len。
            keep_separator (bool): 分割后是否保留分隔符，默认 True。

        Raises:
            ValueError: 当 chunk_overlap >= chunk_size 时抛出。
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator

    def split_text(self, text: str) -> list[str]:
        """将文本分割为 chunk 列表，子类必须实现此方法。

        Args:
            text (str): 待分割的原始文本。

        Returns:
            list[str]: 分割后的 chunk 列表。

        Raises:
            NotImplementedError: 基类未实现，调用时抛出。
        """
        raise NotImplementedError

    def create_documents(self, texts: Iterable[str]) -> list[str]:
        """对多段文本分别分割，返回所有 chunk 的列表。

        Args:
            texts (Iterable[str]): 待分割的多段文本。

        Returns:
            list[str]: 所有文本分割后合并的 chunk 列表。
        """
        chunks: list[str] = []
        for text in texts:
            chunks.extend(self.split_text(text))
        return chunks

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """将过短的片段合并为不超过 chunk_size 的块，并保留 overlap。

        Args:
            splits (list[str]): 已切分的文本片段列表。
            separator (str): 合并片段时使用的连接符。

        Returns:
            list[str]: 合并后满足 chunk_size 限制的文本块列表。
        """
        docs: list[str] = []
        current: list[str] = []
        current_len = 0
        sep_len = self.length_function(separator)

        for split in splits:
            split_len = self.length_function(split)
            # 加上分隔符后的预期长度
            added_len = split_len + (sep_len if current else 0)

            if current_len + added_len > self.chunk_size:
                if current:
                    doc = separator.join(current).strip()
                    if doc:
                        docs.append(doc)
                    # 从末尾保留 overlap
                    while current and (
                        current_len
                        - self.length_function(current[0])
                        - (sep_len if len(current) > 1 else 0)
                        >= self.chunk_overlap
                    ):
                        current_len -= self.length_function(current[0]) + (
                            sep_len if len(current) > 1 else 0
                        )
                        current.pop(0)

            current.append(split)
            current_len += split_len + (sep_len if len(current) > 1 else 0)

        if current:
            doc = separator.join(current).strip()
            if doc:
                docs.append(doc)

        return docs

    @staticmethod
    def _split_with_separator(text: str, separator: str, keep: bool) -> list[str]:
        """用 separator 分割文本，可选择是否保留分隔符。

        Args:
            text (str): 待分割的文本。
            separator (str): 分割符；为空字符串时按单字符分割。
            keep (bool): 是否将分隔符保留并附在前一段末尾。

        Returns:
            list[str]: 分割后的文本片段列表。
        """
        if separator:
            if keep:
                # 保留分隔符：将其附在前一段末尾
                parts = re.split(f"({re.escape(separator)})", text)
                merged: list[str] = []
                i = 0
                while i < len(parts):
                    if i + 1 < len(parts) and parts[i + 1] == separator:
                        merged.append(parts[i] + parts[i + 1])
                        i += 2
                    else:
                        if parts[i]:
                            merged.append(parts[i])
                        i += 1
                return merged
            else:
                return text.split(separator)
        else:
            # separator 为空字符串时按单字符分割
            return list(text)


class CharacterTextSplitter(TextSplitter):
    """按单个分隔符分割文本。

    类似 langchain 的 CharacterTextSplitter。
    """

    def __init__(self, separator: str = "\n\n", **kwargs) -> None:
        """初始化单字符分割器。

        Args:
            separator (str): 分割符，默认为双换行（段落边界）。
            **kwargs: 透传给 TextSplitter 的其余参数。
        """
        super().__init__(**kwargs)
        self.separator = separator

    def split_text(self, text: str) -> list[str]:
        """按分隔符分割文本并合并为合适大小的 chunk。

        Args:
            text (str): 待分割的原始文本。

        Returns:
            list[str]: 分割并合并后的 chunk 列表。
        """
        splits = self._split_with_separator(text, self.separator, self.keep_separator)
        splits = [s for s in splits if s.strip()]
        return self._merge_splits(splits, self.separator if self.keep_separator else " ")


class RecursiveCharacterTextSplitter(TextSplitter):
    """递归字符文本分割器。

    按优先级依次尝试分隔符列表，直到片段足够小。
    默认分隔符顺序与 langchain 一致：段落 → 换行 → 空格 → 字符。
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

    def __init__(
        self,
        separators: list[str] | None = None,
        **kwargs,
    ) -> None:
        """初始化递归字符分割器。

        Args:
            separators (list[str] | None): 按优先级排列的分隔符列表；
                为 None 时使用默认列表 ["\\n\\n", "\\n", " ", ""]。
            **kwargs: 透传给 TextSplitter 的其余参数。
        """
        super().__init__(**kwargs)
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    def split_text(self, text: str) -> list[str]:
        """递归分割文本为满足 chunk_size 的块列表。

        Args:
            text (str): 待分割的原始文本。

        Returns:
            list[str]: 分割后的 chunk 列表。
        """
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """递归地按分隔符优先级分割文本。

        从 separators 中依次选取第一个在文本中出现的分隔符进行分割；
        对仍超过 chunk_size 的片段，用剩余分隔符继续递归分割。

        Args:
            text (str): 当前待分割的文本片段。
            separators (list[str]): 剩余可用的分隔符列表（按优先级排列）。

        Returns:
            list[str]: 递归分割并合并后的 chunk 列表。
        """
        if not text:
            return []

        # 找到第一个在文本中实际出现的分隔符
        separator = separators[-1]  # 兜底：空字符串（逐字符）
        remaining = separators
        for sep in separators:
            if sep == "":
                separator = sep
                remaining = [sep]
                break
            if sep in text:
                separator = sep
                remaining = separators[separators.index(sep) + 1 :]
                break

        splits = self._split_with_separator(text, separator, self.keep_separator)
        splits = [s for s in splits if s.strip()]

        final: list[str] = []
        good: list[str] = []  # 当前累积的待合并小片段

        for s in splits:
            if self.length_function(s) <= self.chunk_size:
                good.append(s)
            else:
                # 先 flush 小片段缓冲，再递归处理大片段，保持原文顺序
                if good:
                    final.extend(self._merge_splits(good, separator if separator else ""))
                    good = []
                if remaining:
                    final.extend(self._split_recursive(s, remaining))
                else:
                    final.append(s)

        # flush 剩余小片段
        if good:
            final.extend(self._merge_splits(good, separator if separator else ""))

        return final


class MarkdownHeaderTextSplitter:
    """按 Markdown 标题层级分割文本。

    返回每个标题块对应的内容与元数据（标题路径）字典列表。
    """

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        strip_headers: bool = True,
    ) -> None:
        """初始化 Markdown 标题分割器。

        Args:
            headers_to_split_on (list[tuple[str, str]] | None): 需要作为分割点的标题标记及其标签，
                格式为 [(标记, 标签), ...]，例如 [("#", "h1"), ("##", "h2")]；
                为 None 时默认使用 h1/h2/h3。
            strip_headers (bool): 是否从内容中去除标题行本身，默认 True。
        """
        # 默认按 #、## 分割
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        # 按 # 数量从多到少排序，确保先匹配最深层级
        self.headers_to_split_on.sort(key=lambda x: len(x[0]), reverse=True)
        self.strip_headers = strip_headers

    def split_text(self, text: str) -> list[dict]:
        """分割 Markdown 文本，按标题层级切分为内容块。

        Args:
            text (str): 待分割的 Markdown 格式文本。

        Returns:
            list[dict]: 每项包含以下字段的字典列表：
                - content (str): 该标题块下的正文内容。
                - metadata (dict[str, str]): 当前块所属的标题路径，键为标签（如 "h1"），值为标题文本。
        """
        lines = text.splitlines()
        chunks: list[dict] = []
        current_content: list[str] = []
        current_metadata: dict[str, str] = {}

        def flush():
            content = "\n".join(current_content).strip()
            if content:
                chunks.append({"content": content, "metadata": dict(current_metadata)})
            current_content.clear()

        for line in lines:
            matched = False
            for marker, label in self.headers_to_split_on:
                pattern = re.compile(rf"^{re.escape(marker)}\s+(.+)$")
                m = pattern.match(line)
                if m:
                    flush()
                    header_text = m.group(1).strip()
                    # 清除比当前层级更深的标题
                    depth = len(marker)
                    current_metadata = {
                        k: v for k, v in current_metadata.items()
                        if len(k) < depth  # 保留比当前浅的层级
                    }
                    current_metadata[label] = header_text
                    if not self.strip_headers:
                        current_content.append(line)
                    matched = True
                    break
            if not matched:
                current_content.append(line)

        flush()
        return chunks

if __name__ == "__main__":
    with open("./skill-guard/scripts/src/test.md", "r", encoding="utf-8") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= 128,
        chunk_overlap= 32,
        separators=["\n\n", "\n", ""],
        keep_separator= True,
    )
    
    # splitter = MarkdownHeaderTextSplitter(
    #     headers_to_split_on=[("#", "h1"), ("##", "h2")],
    #     strip_headers= True,
    # )
    
    chunks = splitter.split_text(text)
    with open("result.txt", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(f"Chunk {i} (length {len(chunk)}):\n{chunk}\n{'-'*40}\n")