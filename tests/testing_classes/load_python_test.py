def testing(**parameters):
    return "time", "values"

def testing_default(test="", **parameters):
    return "time", test

def testing_info(**parameters):
    return "time", "values", {"info": True}