def repeat_fn(f, n):
    """
    Apply function `f` a total of `n` > 0 times.

    @usage:
        `repeat_fn(times_two, 3) 3` 
        Will yield `36`

    @param f: The function to call
    @param n: The total number to call f on x
    """
    if n < 0: 
        raise "[repeat_fn]: Can not call f less than 0 times"
    # Call identity for n = 0 to prevent max-recursion errors
    if n == 0:
        return lambda x: x
    if n == 1:
        return f 
    else: 
        return lambda x: f(repeat_fn(f, n-1)(x))