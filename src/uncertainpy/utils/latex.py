def str_to_latex(text):
    return text.replace("_", "\\_")

def list_to_latex(texts):
    tmp = []
    for txt in texts:
        tmp.append(str_to_latex(txt))

    return tmp