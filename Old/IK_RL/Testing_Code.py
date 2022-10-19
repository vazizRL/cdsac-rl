###Generatoren####
# def square_generator(n):
# 		i = 1
# 		while i <= n:
# 			yield i*i
# 			i += 1
#
# for i_ in square_generator(3):
#     print(i_)


###Iteratoren###
# class Fibonacci:
#
#     def __init__(self,maxN):
#         self.MaxN = maxN
#         self.N = 0
#         self.A = 0
#         self.B = 0
#
#     def __iter__(self):
#         self.N = 0
#         self.A = 0
#         self.B = 1
#         return self
#
#     def __next__(self):
#         if self.N < self.MaxN:
#             self.N += 1
#             self.A, self.B = self.B, self.A + self.B
#             return self.A
#         else:
#             raise StopIteration
#
# for i_ in Fibonacci(5):
#     print(i_)


###Laufbahn Proto + Dynamically creating attributes###
# class Trajectory:
#
#     def __init__(self,*points):
#         keyVal = 97
#         for i in points:
#             self.__dict__[chr(keyVal)] = i
#             keyVal += 1
#
#     def __iter__(self):
#         self.N = 0
#         self.A = 0
#         self.B = 1
#         return self
#
#     def __next__(self):
#         if self.N < self.MaxN:
#             self.N += 1
#             self.A, self.B = self.B, self.A + self.B
#             return self._active_path
#         else:
#             raise StopIteration
#
#
# tracectory = Trajectory()
# def obtain_measurements():

# counter = 0
#
# while True:
#     counter += 1
#     print('Counter is {}'.format(counter))
#     for i in range(30):
#         print(i)
#         if i >= 5:
#             break
#
#     if counter >= 50:
#         break
#
# print("Abgeschlossen")










