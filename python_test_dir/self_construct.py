class Test:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def test_mehtod(self):
        print(self.a, self.b, self.c)
        print('This is a test')

    def update(self, p1, p2, p3):
        self = self.__init__(p1, p2, p3)


if __name__ ==  '__main__':
    test = Test(1, 2, 3)
    test.test_mehtod()
    test.update(4, 5, 6)
    test.test_mehtod()

