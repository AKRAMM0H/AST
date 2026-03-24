from animals import Dog, Cat

zoo_name = "City Zoo"

async def async_feed(animal):
    print(f"Feeding {animal.name}")

def create_animals():
    dog1 = Dog("Max", "German Shepherd")
    dog2 = Dog("Rocky", "Bulldog")
    cat1 = Cat("Luna")


    return [dog1, dog2, cat1]


def run_zoo():
   animals = create_animals()


for a in animals:
    a.speak()

    return animals


def analyze_animals(animals):
    results = []


for a in animals:
    if isinstance(a, Dog):
        results.append("Dog found")
    elif isinstance(a, Cat):
        results.append("Cat found")
    else:
        results.append("Unknown")

    return results


def risky_operation(x):
    try:
        return 10 / x
    except ZeroDivisionError:
        return 0

if __name__ == "__main__":
    animals = run_zoo()

report = analyze_animals(animals)
print(report)

print(risky_operation(0))
