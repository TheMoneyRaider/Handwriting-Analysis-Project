def confidence_correct(text, conf):

    words = text.split()

    corrected = []

    idx = 0

    for word in words:

        length = len(word)

        word_conf = conf[idx:idx+length]

        idx += length + 1

        if word_conf.mean().item() < 0.75:
            corrected.append(spell_correct_word(word))
        else:
            corrected.append(word)

    return " ".join(corrected)





from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2)

sym_spell.load_dictionary(
    "en-80k.txt",
    term_index=0,
    count_index=1
)

def spell_correct_word(word):

    suggestions = sym_spell.lookup(
        word,
        Verbosity.CLOSEST,
        max_edit_distance=2
    )

    if suggestions:
        return suggestions[0].term

    return word


def spell_correct_sentence(text):

    words = text.split()

    corrected = [spell_correct_word(w) for w in words]

    return " ".join(corrected)