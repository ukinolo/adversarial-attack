import keras
import matplotlib.pyplot as plt
import numpy as np
import adversarial_attack.galib as galib

def _encode_input(img) -> np.array:
    return np.array(img).flatten()

def _decode_input(array : np.array, shape):
    return array.reshape(shape)

class AdversarialAttack:
    def __init__(self, model_path: str):
        #Parameters
        self.change_cap = 0.15
        self.alpha = 0.5
        self.beta = 0.8

        self.model = keras.models.load_model(model_path)

        config = self.model.get_config()
        batch_shape = config["layers"][0]["config"]["batch_shape"]
        self.input_shape = (batch_shape[1], batch_shape[2], batch_shape[3])
        self.input_len = self.input_shape[0] * self.input_shape[1] * self.input_shape[2]
        
        self.input = np.zeros(self.input_shape)
        self.flatten_input = _encode_input(self.input)
        self.correct_category_index = 0
    
        self.fitness0 = galib.Fitness(lambda x: self._reduce_correct_confidence_fitness(x), self.input_len)
        self.fitness1 = galib.Fitness(lambda x: self._increase_wrong_confidence_fitness(x), self.input_len)
        self.fitness2 = galib.Fitness(lambda x: self._diversity_fitness(x), self.input_len)
        self.fitnesses = [self.fitness0, self.fitness1, self.fitness2]

        self.selection = galib.Selection('tournament', tournament_size=5)
        self.crossover = galib.Crossover('sbx', distribution_index=1.0, probability_of_crossover=0.05)
        self.mutation = galib.Mutation('normal', 0.65, 0.02, standard_deviation=1.0)
        self.replacement = galib.Replacement('elitism', elitism_width=0.02)

        self.ga_config = galib.GA(self.fitnesses, self.selection, self.crossover, self.replacement, self.mutation, change_fitness_function=self._get_fitness_to_use)

    def attack(self, input, correct_category_index, output_path='./result'):
        # Check if the input is valid
        self.correct_category_index = correct_category_index
        self.input = input
        self.flatten_input = _encode_input(self.input).astype("float32") / 255
        
        # Start the ga
        self.ga_config.initialize_population_interval(100, interval_min = -self.change_cap, interval_max = self.change_cap)
        best_individuals, _, _, _, _ = self.ga_config.run_mf(50, 500, run_times=1, early_stopping_rounds=15, verbose=10)

        # Save the attack to the file
        result = _decode_input(self.flatten_input + best_individuals[0], self.input_shape)
        pred = self.model.predict(np.array([result]), verbose=0)[0]
        print(f"Correct category = {self.correct_category_index}, result category = {np.where(pred == max(pred))[0]}, with prediction {round(max(pred), 3)}")
        plt.imshow(result, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    def _get_fitness_to_use(self, generation_best_individual, generation_num):
        t = generation_best_individual + self.flatten_input
        t = _decode_input(t, self.input_shape)
        t = np.array([t])
        preds = self.model.predict(t, verbose=0)
        preds = preds[0]
        correct_accuracy = preds[self.correct_category_index]
        preds = np.delete(preds, self.correct_category_index)
        if correct_accuracy - max(preds) > self.alpha and generation_num < 250:
            return 0 #Explore, lower correct category
        else:
            if max(preds) - correct_accuracy > self.beta:
                return 2 #Reduce the difference from the original
            else: return 1 #Increase the wrongness confidence

    def _reduce_correct_confidence_fitness(self, x) -> float:
        t = self.flatten_input.copy()
        for i in range(len(x)):
            if x[i] > self.change_cap:
                x[i] = self.change_cap
            elif x[i] < -self.change_cap:
                x[i] = -self.change_cap
            t[i] += x[i]
            if t[i] > 1:
                x[i] -= t[i] - 1
                t[i] = 1
            elif t[i] < 0:
                x[i] -= t[i]
                t[i] = 0
        t = _decode_input(t, self.input_shape)
        t = np.array([t])
        preds = self.model.predict(t, verbose=0)
        preds = preds[0]
        fitness_value = -preds[self.correct_category_index]
        return fitness_value

    def _increase_wrong_confidence_fitness(self, x) -> float:
        s = 0
        t = self.flatten_input.copy()
        for i in range(len(x)):
            if x[i] > self.change_cap:
                x[i] = self.change_cap
            elif x[i] < -self.change_cap:
                x[i] = -self.change_cap
            t[i] += x[i]
            if t[i] > 1:
                x[i] -= t[i] - 1
                t[i] = 1
            elif t[i] < 0:
                x[i] -= t[i]
                t[i] = 0
            s += abs(x[i])
        t = _decode_input(t, self.input_shape)
        t = np.array([t])
        preds = self.model.predict(t, verbose=0)
        preds = preds[0]
        preds = np.delete(preds, self.correct_category_index)
        fitness_value = max(preds)
        fitness_value -= s/800
        return fitness_value

    def _diversity_fitness(self, x) -> float:
        s = 0
        t = self.flatten_input.copy()
        for i in range(len(x)):
            if x[i] > self.change_cap:
                x[i] = self.change_cap
            elif x[i] < -self.change_cap:
                x[i] = -self.change_cap
            t[i] += x[i]
            if t[i] > 1:
                x[i] -= t[i] - 1
                t[i] = 1
            elif t[i] < 0:
                x[i] -= t[i]
                t[i] = 0
            s += abs(x[i])
        t = _decode_input(t, self.input_shape)
        t = np.array([t])
        preds = self.model.predict(t, verbose=0)
        preds = preds[0]
        preds = np.delete(preds, self.correct_category_index)
        fitness_value = max(preds)
        fitness_value -= s/150
        return fitness_value