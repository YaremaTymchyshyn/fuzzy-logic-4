import sys
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def build_lighting_controller():
    external = ctrl.Antecedent(np.linspace(0, 1000, 1001), 'external')
    presence = ctrl.Antecedent(np.linspace(0, 10, 11), 'presence')
    time_of_day = ctrl.Antecedent(np.linspace(0, 24, 25), 'time')

    brightness = ctrl.Consequent(np.linspace(0, 100, 101), 'brightness')

    external['dark'] = fuzz.trapmf(external.universe, [0, 0, 200, 400])
    external['medium'] = fuzz.trimf(external.universe, [300, 500, 700])
    external['bright'] = fuzz.trapmf(external.universe, [600, 800, 1000, 1000])

    presence['none'] = fuzz.trimf(presence.universe, [0, 0, 1])
    presence['few'] = fuzz.trimf(presence.universe, [1, 3, 5])
    presence['many'] = fuzz.trimf(presence.universe, [4, 7, 10])

    time_of_day['night'] = fuzz.trapmf(time_of_day.universe, [0, 0, 4, 6])
    time_of_day['morning'] = fuzz.trimf(time_of_day.universe, [5, 8, 12])
    time_of_day['afternoon'] = fuzz.trimf(time_of_day.universe, [11, 15, 18])
    time_of_day['evening'] = fuzz.trapmf(time_of_day.universe, [17, 19, 24, 24])

    brightness['low'] = fuzz.trimf(brightness.universe, [0, 0, 30])
    brightness['medium'] = fuzz.trimf(brightness.universe, [20, 50, 80])
    brightness['high'] = fuzz.trimf(brightness.universe, [70, 100, 100])

    rules = [
        ctrl.Rule(external['dark'] & (presence['few'] | presence['many']), brightness['high']),
        ctrl.Rule(external['dark'] & time_of_day['morning'] & presence['few'], brightness['medium']),
        ctrl.Rule(external['medium'] & presence['many'], brightness['medium']),
        ctrl.Rule(external['bright'] | presence['none'], brightness['low']),
        ctrl.Rule(time_of_day['night'] & (presence['few'] | presence['many']), brightness['medium']),
        ctrl.Rule(time_of_day['evening'] & external['medium'], brightness['high']),
        ctrl.Rule(time_of_day['afternoon'] & external['dark'], brightness['medium']),
        ctrl.Rule(time_of_day['morning'] & external['bright'], brightness['low']),
        ctrl.Rule(time_of_day['evening'] & presence['many'], brightness['high']),
    ]

    lighting_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(lighting_ctrl)


def get_input_value(prompt, min_val, max_val):
    while True:
        user = input(prompt)
        if user.lower() == 'e':
            print("Exiting...")
            sys.exit(0)
        try:
            val = float(user)
            if not (min_val <= val <= max_val):
                raise ValueError
            return val
        except ValueError:
            print(f"Invalid value. Please try again.")


def main():
    sim = build_lighting_controller()
    print("Smart Lightning Fuzzy Controller")
    print("Press 'e' to exit the program\n")
    while True:
        ext = get_input_value("External lightning (0–1000 lux): ", 0, 1000)
        pres = get_input_value("Amount of people (0–10): ", 0, 10)
        t = get_input_value("Time of day (0–24): ", 0, 24)
        sim.input['external'] = ext
        sim.input['presence'] = pres
        sim.input['time'] = t
        sim.compute()
        print(f"Estimated brightness: {sim.output['brightness']:.2f}%\n")


if __name__ == '__main__':
    main()
