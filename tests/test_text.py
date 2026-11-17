"""Words in Indian scripts keep their vowel signs in captions, laugh-word matching and file names."""
import clipper

HAHA = "हाहा"            # हाहा: the vowel sign ा is a combining mark, not \w


def test_caption_keeps_vowel_signs():
    ass = clipper.build_ass([clipper.Word(0.0, 0.5, HAHA + "!")], 0.0, 1.0, 1080, 1920, True)
    assert HAHA in ass


def test_hindi_laugh_word_matches():
    token = clipper.laugh_token(HAHA + "!")
    assert token == HAHA
    assert clipper.LAUGH_WORDS.match(token)


def test_slug_keeps_vowel_signs():
    assert clipper.slugify("\U0001F534 LIVE " + HAHA + " \U0001F602") == "LIVE_" + HAHA
