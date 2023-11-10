class Test:
    def __init__(self, inp1, inp2):
        self.inp1 = inp1
        self.inp2 = inp2

    def __repr__(self):
        arg_string = f'The attributes are {self.inp1, self.inp2}'
        return arg_string


if __name__ == '__main__':
    test = Test(inp1=1, inp2=2)
    print(repr(test))





