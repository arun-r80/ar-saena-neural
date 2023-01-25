import math


def one_more_enclosing():
    v = "This is enclosing"
    global arun

    def test_scope():
        def do_local():
            do_local()

    def global_in_nonlocal():
        global arun
        arun = "Assignment in global_in_nonlocal"

    test_scope()
    global_in_nonlocal()
    arun = " In upppermost non local"


def add(count):
    count = count + 1


class Complex:
    count = 3
    f = add

    def __init__(self, real=2, conjugate=3):
        self.real_component = real
        self.complex_component = conjugate
        self.set_type()

    def set_type(self):
        self.type = "Complex"

    def get_magnitude(self):
        return math.sqrt(math.pow(self.real_component, 2) + math.pow(self.complex_component, 2))


class Vector(Complex):
    real_component = 2
    complex_component = 3

    def set_type(self):
        self.type = "Vector"
        print("Inside set type function", self.real_component)



class Employee:

    def __init__(self, empid, first_name, last_name, role):
        self.empid = empid
        self.first_name = first_name
        self.last_name = last_name
        self.role = role
        self.status = "contract"

    def make_perm(self):
        self.status = "permanent"

class EmployeeList:

    def __init__(self, iter):
        self.employeelist = []
        self.index = 0;
        for emp in iter:
            self.employeelist.append(emp)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index <= (len(self.employeelist) - 1):
            nxt_iter = self.employeelist[self.index]
            self.index += 1
            return nxt_iter
        else:
            raise StopIteration

emp1 = Employee('e1', "John", "Doe", "PM")
emp2 = Employee("e2", "Visa", "Kathiresan", "BA")
emp3 = Employee("e3", "Anand", "Kannan", "BA")
emp4 = Employee("e4", "Debra", "Symons", "Senior BA")

employeeList = EmployeeList([emp1, emp2, emp3, emp4])

for employee in employeeList:
    print("Employee Id: ", employee.empid, "Employee First Name: ", employee.first_name)



c = Complex(2, 3)
d = Vector(2, 3)
# print("Attribute in base class ", c.type)
# print("Attribute in derived class", d.type)

a = Complex(2, 3)
b = Complex(1, 2)
# Complex.get_magnitude()
a1 = a.get_magnitude
print("__self__ attribute of method function",d.set_type.__self__.real_component)

print("Calling function method")
Vector.set_type(Vector())



