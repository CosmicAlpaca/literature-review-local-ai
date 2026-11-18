def build_query_variants(topic: str) -> list[str]:
    topic = topic.strip()
    variants = [
        topic,
        f'"{topic}"',
        f"{topic} review",
        f"{topic} survey",
    ]
    return list(dict.fromkeys(v for v in variants if v))