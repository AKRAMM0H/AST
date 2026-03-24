import os
import sys
from pathlib import Path
import ast
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tiktoken

MAX_RETRIES = 3
DEBUG = True
NUM_ITER = 500


class Animal:
    """Animal is a class represent the animal"""
    pass

class Dog(Animal):
    """Dog is class represent the action of dogs"""
    def __init__(self, name):
        self.name = name

    def bark(self):
        """bark is a function to represent the action of bark"""
        print(f"{self.name} says Woof!")

    def fetch(self, item):
        return f"{self.name} fetched {item}"

    def sleep(self):
        print("Zzz...")

    def eat(self, food):
        print(f"{self.name} eats {food}")

    def run(self, distance):
        return f"{self.name} ran {distance}km"

    def play(self, game):
        print(f"{self.name} plays {game}")

    def greet(self, other_dog):
        other_dog.bark()

    def train(self, command):
        print(f"{self.name} learned {command}")

    def vet_visit(self):
        self.eat("medicine")
        self.sleep()

def train_dog(dog, commands):
    for command in commands:
        dog.train(command)
        
async def fetch_dog_data(dog_name):
    """Fetch dog data from an API."""
    response = await get_api_response(dog_name)
    return response

def main():
    dog = Dog("Rex")
    dog.bark()
    train_dog(dog, ["sit", "stay"])