from release.providers.msg_content import (
    NormalizedMessage,
    Part,
    extract_text,
    normalize_messages,
    normalize_parts,
)


def test_normalize_parts_handles_text_and_images() -> None:
    content = [
        {"type": "text", "text": "hello"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
        {
            "type": "input_image",
            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
        },
    ]

    parts = normalize_parts(content)

    assert isinstance(parts[0], Part.Text)
    assert isinstance(parts[1], Part.UrlImage)
    assert isinstance(parts[2], Part.DataImage)
    assert parts[2].media_type == "image/png"


def test_extract_text_from_multipart_content() -> None:
    content = [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": "https://example.com/a.png"},
        {"type": "text", "text": "second"},
    ]

    assert extract_text(content) == "firstsecond"


def test_normalize_messages_merges_system_and_maps_roles() -> None:
    messages = [
        {"role": "system", "content": "sys-a"},
        {"role": "developer", "content": [{"type": "text", "text": "sys-b"}]},
        {"role": "assistant", "content": "I can help."},
        {"role": "user", "content": [{"type": "text", "text": "Do it."}]},
        {
            "role": "tool",
            "content": "ignored role still maps to user when content exists",
        },
    ]

    system_prompt, normalized = normalize_messages(messages)

    assert system_prompt == "sys-a\n\nsys-b"
    assert normalized == [
        NormalizedMessage(role="assistant", parts=(Part.Text(text="I can help."),)),
        NormalizedMessage(role="user", parts=(Part.Text(text="Do it."),)),
        NormalizedMessage(
            role="user",
            parts=(
                Part.Text(text="ignored role still maps to user when content exists"),
            ),
        ),
    ]
