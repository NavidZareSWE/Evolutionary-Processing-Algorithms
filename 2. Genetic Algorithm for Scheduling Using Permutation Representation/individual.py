# ############################### Individual Class #############################

class Individual:
    # Class variable to track the next available ID
    _next_id = 0

    def __init__(self, chromosome, generationBirth=0):
        self._chromosome = chromosome
        self._age = 0
        self._generationBirth = generationBirth
        self._fitness_calculated = False
        self._fitness = None

        # Assign unique ID and increment the class counter
        self._id = Individual._next_id
        Individual._next_id += 1

    def get_chromosome(self):
        return self._chromosome

    def set_chromosome(self, value):
        self._chromosome = value
        self._fitness_calculated = False  # For chromosome changes

    def get_fitness(self):
        return self._fitness

    def set_fitness(self, fitness):
        self._fitness = fitness
        self._fitness_calculated = True

    def is_fitness_calculated(self):
        return self._fitness_calculated

    def get_age(self):
        return self._age

    def get_generation(self):
        return self._generationBirth

    def set_generation(self, value):
        self._generationBirth = value

    def increment_generation(self):
        self._generationBirth += 1

    def increment_age(self):
        self._age += 1

    def get_id(self):
        return self._id
