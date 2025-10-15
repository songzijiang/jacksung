def round2str(number, digits=2):
    number = round(number, digits)
    if digits == 0:
        return str(int(number))
    else:
        return f"{number:.{digits}f}"
