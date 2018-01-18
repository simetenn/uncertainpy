def str_to_latex(text):
    r"""
    Convert a string to latex format, replace "_" with "\\_".

    Parameters
    ----------
    text : str
        Text to convert to latex format.

    Returns
    -------
    text : str
        Text converted to latex format.
    """
    return text.replace("_", "\\_")


def list_to_latex(texts):
    """
    Convert a list of strings to latex format, replace "_" with "\\_" in each
    string.

    Parameters
    ----------
    texts : list
        List of strings to convert to latex format.

    Returns
    -------
    texts : str
        List of strings converted to latex format.

    See also
    --------
    uncertainpy.utils.str_to_latex : Convert a string to latex
    """
    tmp = []
    for txt in texts:
        tmp.append(str_to_latex(txt))

    return tmp