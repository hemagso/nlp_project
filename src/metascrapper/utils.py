def safe_cast(v, to_type, default=None):
    """safe_cast
    Exception safe casting function
    :param v: Value to be cast
    :param to_type: Type to be cast into
    :param default: Default value in case of failure
    :return: Casted value, or default value in failure
    """
    try:
        return to_type(v)
    except (ValueError, TypeError):
        return default