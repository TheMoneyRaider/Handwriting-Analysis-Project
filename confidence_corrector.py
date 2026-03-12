def confidence_correct(text, conf):

    words = text.split()

    corrected = []
    idx = 0

    for word in words:

        length = len(word)

        if idx + length > len(conf):
            corrected.append(word)
            continue

        word_conf = conf[idx:idx+length]
        idx += length

        if word_conf.mean().item() < 0.6:
            corrected.append(spell_correct_word(word))
        else:
            corrected.append(word)

        idx += 1  # skip space

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

    if suggestions and suggestions[0].distance > 0:
        return suggestions[0].term

    return word