class Debug(object):
    def __init__(self):
        self.enabled = False

    def __bool__(self):
        return self.enabled

    def __getattr__(self, key):
        result = Debug()
        object.__setattr__(self, key, result)
        return result

debug = Debug()