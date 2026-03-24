import math

GLOBAL_VAR = 42

class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclasses must implement this")

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed


    def bark(self):
        print(f"{self.name} says Woof!")

    def speak(self):
        self.bark()


class Cat(Animal):
    def speak(self):
        print(f"{self.name} says Meow!")

def compute_circle_area(radius):
    return math.pi * radius ** 2

def process_animals(animals):
    for animal in animals:
        try:
            animal.speak()
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    dog = Dog("Buddy", "Labrador")
    cat = Cat("Whiskers")

process_animals([dog, cat])

area = compute_circle_area(5)
print("Area:", area)

