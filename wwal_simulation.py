"""
@Author: Johannes Spieler
@Date: 25.04.2022
"""
import csv
import numpy.random
from dataclasses import dataclass, field
from logging import error
from math import exp
from pathlib import Path
from typing import List, Dict
from matplotlib import pyplot as plt


@dataclass
class Sim:
    products: List[int] = field(init=False, default_factory=list)
    tasks: List[int] = field(init=False, default_factory=list)
    stations: List[int] = field(init=False, default_factory=list)
    workers: List[int] = field(init=False, default_factory=list)

    tasks_of_station: Dict[int, List[int]] = field(init=False, default_factory=dict)
    tasks_of_worker: Dict[int, List[int]] = field(init=False, default_factory=dict)
    tasks_of_product: Dict[int, List[int]] = field(init=False, default_factory=dict)

    task_start_times: Dict[int, float] = field(init=False, default_factory=dict)
    task_queue_durations: Dict[int, float] = field(init=False, default_factory=dict)
    task_finish_times: Dict[int, float] = field(init=False, default_factory=dict)

    precedences: Dict[int, List[int]] = field(init=False, default_factory=dict)

    break_starts_of_worker: Dict[int, List[float]] = field(init=False, default_factory=dict)
    break_ends_of_worker: Dict[int, List[float]] = field(init=False, default_factory=dict)

    tasks_during_break_time: Dict[int, List[float]] = field(init=False, default_factory=dict)

    actual_task_durations: Dict[int, float] = field(init=False, default_factory=dict)

    average_task_durations: Dict[int, float] = field(init=False, default_factory=dict)
    variance_of_task_completion_times: Dict[int, float] = field(init=False, default_factory=dict)
    straining_levels_of_stations: Dict[int, float] = field(init=False, default_factory=dict)

    beta: float = field(init=False)
    mu: float = field(init=False)
    lam: float = field(init=False)
    duration: int = field(init=False)

    def assign_task_to_worker(self, worker: int, task: int):
        if worker not in self.tasks_of_worker:
            self.tasks_of_worker[worker] = []
        self.tasks_of_worker[worker].append(task)

    def get_last_assignment_of_workers(self):
        return {worker: max([self.task_finish_times[x] for x in self.tasks_of_worker[worker]]) for worker in
                self.workers}

    def add_new_product(self):
        product_no: int = len(self.products)
        self.products.append(product_no)
        offset = len(self.tasks)
        tasks = []
        for station_no in range(len(self.stations)):
            task = station_no + offset
            self.precedences[task] = [x for x in tasks]
            if station_no not in self.tasks_of_station:
                self.tasks_of_station[station_no] = []
            self.tasks_of_station[station_no].append(task)
            tasks.append(task)
            self.actual_task_durations[task] = give_normal_distribution(self.average_task_durations[station_no],
                                                                        self.variance_of_task_completion_times[
                                                                            station_no])
        self.tasks_of_product[product_no] = tasks
        self.tasks += tasks
        return product_no

    def start_up(self):
        for worker in self.workers:
            self.worker_to_next_task(worker)

    def give_task_start_time(self, task: int, st: float):
        if task in self.task_start_times:
            error('start time already declared')
        self.task_start_times[task] = st

    def give_task_finish_time(self, task: int, ft: float):
        if task in self.task_finish_times:
            error('finish time already declared!')
        self.task_finish_times[task] = ft

    def get_earliest_start_at_station(self, station):
        if station not in self.tasks_of_station:
            return 0
        if len([each for each in self.tasks_of_station[station] if
                each in self.task_finish_times]) == 0:
            return 0
        return max(
            [self.task_finish_times[each] for each in self.tasks_of_station[station] if
             each in self.task_finish_times])

    def give_worker_first_product(self, worker, current_time: float):
        product_no = self.add_new_product()
        task = min(self.tasks_of_product[product_no])
        self.tasks_of_worker[worker] = []
        self.tasks_of_worker[worker].append(task)
        station_time = self.get_earliest_start_at_station(0)
        delay = station_time - current_time
        if delay > 0:
            self.task_queue_durations[task] = delay
        self.give_task_start_time(task, station_time)
        self.give_task_finish_time(task, self.actual_task_durations[task] + delay)

    def task_on_end_of_line(self, task: int):
        return task in self.tasks_of_station[len(self.stations) - 1]

    def worker_to_next_task(self, worker: int):
        if worker not in self.tasks_of_worker:
            self.give_worker_first_product(worker, 0)
            return
        last_task = max(self.tasks_of_worker[worker])
        # check if task has been added:
        if self.task_on_end_of_line(last_task):
            product_number = self.add_new_product()
            new_task = min(self.tasks_of_product[product_number])
        else:
            new_task = last_task + 1
        # make sure all predecessors have been finished
        if len(self.precedences[new_task]):
            youngest_predecessor = max(self.precedences[new_task], key=lambda x: self.task_finish_times[x])
        else:
            youngest_predecessor = last_task
        prev_task_finish = self.task_finish_times[youngest_predecessor]
        station_number = new_task % len(self.stations)
        # since all tasks of P1 have lower numbers than all tasks of P2:
        tasks_from_station = [z for z in self.tasks_of_station[station_number] if z < new_task]
        if len(tasks_from_station) != 0:
            station_predecessor = max(tasks_from_station)
        else:
            # -1 equivalent to non existent
            station_predecessor = -1
            pass
        if station_predecessor not in self.task_finish_times:
            prev_station_finish = 0
        else:
            prev_station_finish = self.task_finish_times[station_predecessor]

        if prev_station_finish > prev_task_finish:
            queue_time = prev_station_finish - prev_task_finish
            earliest_start = prev_station_finish
        else:
            queue_time = 0
            earliest_start = prev_task_finish
        task_duration = self.actual_task_durations[new_task]
        for break_start, break_finish in zip(self.break_starts_of_worker[worker], self.break_ends_of_worker[worker]):
            start_in_break = between(break_start, earliest_start - queue_time, break_finish)
            finish_in_break = between(break_start, earliest_start + task_duration, break_finish)
            if start_in_break:
                task_delay = break_finish - earliest_start - queue_time
                earliest_start += task_delay
                self.task_queue_durations[new_task] = queue_time
                self.assign_task_to_worker(worker, new_task)
                self.give_task_start_time(new_task, earliest_start)
                self.give_task_finish_time(new_task, earliest_start + task_duration)
                return
            elif finish_in_break:
                # slice 2,4
                part_one_duration = break_start - earliest_start
                part_two_duration = task_duration - part_one_duration
                self.tasks_during_break_time[new_task] = [break_start, break_finish]
                if station_predecessor in self.tasks_during_break_time:
                    queue_time -= self.tasks_during_break_time[station_predecessor][1] - \
                                  self.tasks_during_break_time[station_predecessor][0]
                self.task_queue_durations[new_task] = queue_time
                self.assign_task_to_worker(worker, new_task)
                self.give_task_start_time(new_task, earliest_start)
                self.give_task_finish_time(new_task,
                                           earliest_start + part_one_duration + break_finish - break_start
                                           + part_two_duration)

                return
            elif (earliest_start - queue_time < break_start and
                  earliest_start + task_duration > break_finish):
                self.tasks_during_break_time[new_task] = [break_start, break_finish]
                self.task_queue_durations[new_task] = queue_time
                self.assign_task_to_worker(worker, new_task)
                self.give_task_start_time(new_task, earliest_start)
                self.give_task_finish_time(new_task, earliest_start + task_duration)
                return

        self.task_queue_durations[new_task] = queue_time
        self.assign_task_to_worker(worker, new_task)
        self.give_task_start_time(new_task, earliest_start)
        self.give_task_finish_time(new_task, earliest_start + task_duration)

    def calculate_fatigue_in_timeframe(self, worker: int, t_start: float, t_finish: float, start_fatigue: float):

        tasks = [task for task in self.tasks_of_worker[worker] if
                 self.task_start_times[task] - self.task_queue_durations.get(task, 0) < t_finish and
                 self.task_finish_times[
                     task] > t_start]
        fatigue = start_fatigue
        for task in tasks:
            if task in self.task_queue_durations:
                start = max(self.task_start_times[task] - self.task_queue_durations[task], t_start)
            else:
                start = max(self.task_start_times[task], t_start)
            finish = min(self.task_finish_times[task], t_finish)
            if task in self.tasks_during_break_time:

                break_start = max(self.tasks_during_break_time[task][0], t_start)
                break_finish = min(self.tasks_during_break_time[task][1], t_finish)
                if break_start < break_finish:
                    fatigue = do_fatigue_break(fatigue, self.mu, break_finish - break_start)
            fatigue = do_fatigue_work(fatigue, self.lam, finish - start)
        return fatigue


def between(a: float, b: float, c: float):
    return a <= b < c


def give_normal_distribution(expected_time_in_seconds: float, variance: float):
    return numpy.random.normal(expected_time_in_seconds, variance)


def do_fatigue_work(old_fatigue: float, lam: float, duration: float):
    return old_fatigue + ((1 - old_fatigue) * (1 - exp(lam / 60 * duration)))


def do_fatigue_break(old_fatigue: float, mu: float, duration: float):
    return old_fatigue * exp(mu / 60 * duration)


def calc_error(beta: float, fatigue: float):
    return pow(beta * fatigue, 6)


def save_schedule_to_png(sim: Sim, duration: int, title_prefix: str = '', title_appendix: str = ''):
    plt.clf()
    plt.cla()
    fig, ax = plt.subplots()
    ax.set_xlim(0, duration)
    ax.set_ylim(0, len(sim.workers))
    ax.grid(True)
    ax.set_xlabel('time')
    ax.set_ylabel('worker')
    ax.set_yticks(range(0, len(sim.workers), 1))
    ax.grid(True)
    for index, worker in enumerate(sim.workers):
        for task in sim.tasks_of_worker[worker]:
            finish = sim.task_finish_times[task]
            start = sim.task_start_times[task]
            if task in sim.tasks_during_break_time:
                break_start = sim.tasks_during_break_time[task][0]
                break_finish = sim.tasks_during_break_time[task][1]
                if task in sim.task_queue_durations:
                    queue = sim.task_queue_durations[task]
                    queue_start = start - queue
                    if queue_start < break_start < start:
                        # split queue
                        first_queue_duration = break_start - queue_start
                        second_queue_duration = queue - first_queue_duration
                        ax.broken_barh([(queue_start, first_queue_duration), [break_finish, second_queue_duration]],
                                       (index, 1), facecolors='tab:red')

                        ax.broken_barh([(start, finish - start)], (index, 1))
                    else:
                        # split work
                        ax.broken_barh([(start - queue, break_start - start + queue)],
                                       (index, 1), facecolors='tab:red')
                        tail_time = finish - break_start
                        ax.broken_barh([(start, break_start - start), (break_finish, tail_time)], (index, 1))
                else:
                    tail_time = finish - break_start
                    ax.broken_barh([(start, break_start - start), (break_finish, tail_time)], (index, 1))
            elif task not in sim.task_queue_durations:
                ax.broken_barh([(start, finish - start)], (index, 1))
            else:
                queue = sim.task_queue_durations[task]
                ax.broken_barh([(start - queue, queue)], (index, 1), facecolors='tab:red')
                ax.broken_barh([(start, finish - start)], (index, 1))
    plt.savefig(f'{title_prefix}_schedule_{title_appendix}.png')
    plt.close(fig)


def calculate_worker_fatigue_levels(sim: Sim):
    duration = sim.duration
    total_steps = 300
    worker_fatigues = {worker: {0: 0.0} for worker in sim.workers}
    steps = [int(x) for x in range(0, duration, int(duration / total_steps))]
    last_step = 0

    for step in steps:
        for worker in sim.workers:
            last_fatigue = worker_fatigues[worker][last_step]
            new_fatigue = sim.calculate_fatigue_in_timeframe(worker, last_step, step, last_fatigue)
            worker_fatigues[worker][step] = new_fatigue
        last_step = step
    return worker_fatigues


def calculate_avg_error_rate(worker_fatigues):
    error_sum = 0
    total = 0
    for worker in worker_fatigues:
        for time in worker_fatigues[worker]:
            error_sum += calc_error(0.6, worker_fatigues[worker][time])
            total += 1
    return error_sum / total


def save_data_to_csv(data_list: List[float]):
    with open('data_save.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(data_list)


def eval_worker_utilization(sim: Sim):
    task_time_total = 0
    queue_time_total = 0
    for task in sim.tasks:
        if task in sim.task_queue_durations:
            queue_time_total += sim.task_queue_durations[task]
        task_time_total += sim.actual_task_durations[task]

    return task_time_total / (task_time_total + queue_time_total)


def save_fatigue_and_error_function_to_png(fatigue_save, title_prefix: str, title_appendix: str):
    plt.clf()
    plt.cla()

    fig, ax = plt.subplots()
    ax.set_xlabel('time')
    ax.set_ylabel('fatigue')
    for worker in fatigue_save:
        keys = [key for key in fatigue_save[worker]]
        ax.plot([key for key in keys], [fatigue_save[worker][key] for key in keys])
    fig.savefig(f'{title_prefix}_fatigue_function_{title_appendix}.png')
    plt.close(fig)
    ax.cla()
    ax.set_ylabel('human error probability')
    ax.set_xlabel('time')
    for worker in fatigue_save:
        keys = [key for key in fatigue_save[worker]]
        plt.plot([key for key in keys], [calc_error(0.6, fatigue_save[worker][key]) for key in keys])
    fig.savefig(f'{title_prefix}_error_function_{title_appendix}.png')
    plt.close(fig)


def run_until(sim: Sim, until=32000):
    while True:
        worker_finishes = sim.get_last_assignment_of_workers()
        running_finishes = {w: worker_finishes[w] for w in worker_finishes}
        min_finish = min(running_finishes, key=lambda x: running_finishes[x])
        if worker_finishes[min_finish] > until:
            break
        sim.worker_to_next_task(min_finish)

    return sim


def run_simulation(workers: int, stations: int, beta: float, mu: float, lam: float, duration: int, rest_schedule: [],
                   expected_durations: {}, variances: [], runs: int, png_title_prefix: str = ''):
    avg_error_total = 0
    worker_utilization_total = 0
    for run in range(runs):
        sim = Sim()
        sim.beta = beta
        sim.mu = mu
        sim.lam = lam
        sim.workers = [x for x in range(workers)]
        sim.stations = [x for x in range(stations)]
        sim.variance_of_task_completion_times = variances
        sim.average_task_durations = expected_durations
        sim.break_ends_of_worker = {x: [each[1] for each in rest_schedule] for x in range(workers)}
        sim.break_starts_of_worker = {x: [each[0] for each in rest_schedule] for x in range(workers)}
        sim.duration = duration
        sim.start_up()

        sim = run_until(sim, duration)

        save_schedule_to_png(sim, duration, title_prefix=png_title_prefix, title_appendix=f'run{run}')
        worker_fatigue_levels_over_time = calculate_worker_fatigue_levels(sim)
        avg_error_rate_of_run = calculate_avg_error_rate(worker_fatigue_levels_over_time)
        avg_error_total += avg_error_rate_of_run
        worker_utilization_total += eval_worker_utilization(sim)
        save_fatigue_and_error_function_to_png(worker_fatigue_levels_over_time, png_title_prefix, f'run{run}')
    save_data_to_csv([avg_error_total / runs, worker_utilization_total / runs])


def main():
    Path('./figs/').mkdir(exist_ok=True)
    # mu and lam taken from !!!!!!
    mu = -0.0096
    lam = -0.0096
    beta = 0.6
    workers = 11
    stations = 10
    # stations set up according to Al-Zuheri et al.
    expected_durations = [
        105,
        69,
        100,
        82,
        115,
        121,
        150,
        102,
        93,
        82
    ]

    variances = [0.1 * x for x in expected_durations]
    work_day_in_seconds = 8 * 60 * 60
    take_break_in_middle_schedule = [(4 * 60 * 60, 4.5 * 60 * 60)]
    take_break_at_end_schedule = [(7.5 * 60 * 60, 8 * 60 * 60)]
    take_small_breaks_schedule = [(3 * 60 * 60, 3.25 * 60 * 60), (6 * 60 * 60, 6.25 * 60 * 60)]

    no_break_example_break_schedule = []

    givi_example_break_schedule = [
        (125 * 60, 135 * 60),
        (255 * 60, 300 * 60),
        (420 * 60, 430 * 60),
    ]

    runs = 50
    print('starting no break simulation ...')
    run_simulation(workers, stations, beta, mu, lam, (530 - 10 - 45 - 10) * 60, no_break_example_break_schedule,
                   expected_durations, variances, 1, './figs/no_break')

    print('starting givi simulation ...')
    run_simulation(workers, stations, beta, mu, lam, 530 * 60, givi_example_break_schedule,
                   expected_durations, variances, 1, './figs/givi')

    print('starting first simulation ...')
    run_simulation(workers, stations, beta, mu, lam, work_day_in_seconds, take_break_in_middle_schedule,
                   expected_durations, variances, runs, './figs/middle_break')

    print('starting second simulation ...')
    run_simulation(workers, stations, beta, mu, lam, work_day_in_seconds, take_break_at_end_schedule,
                   expected_durations, variances, runs, './figs/end_break')

    print('starting third simulation ...')
    run_simulation(workers, stations, beta, mu, lam, work_day_in_seconds, take_small_breaks_schedule,
                   expected_durations, variances, runs, './figs/small_breaks')


if __name__ == '__main__':
    main()
