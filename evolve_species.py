import datetime
import signal
import subprocess

import experiments
import kernel


user_requested_abort = False


def handle_sigint(sig, frame):
    global user_requested_abort
    if user_requested_abort:
        print('Caught Ctrl+C again, terminating immediately.')
        exit(0)
    else:
        user_requested_abort = True
        print('Caught Ctrl+C, will terminate after current trial ends.')


def format_time(time_in_seconds):
    result = str(datetime.timedelta(seconds=int(time_in_seconds)))
    if result.startswith('0:'):
        result = result[2:]
    return result


def print_status_summary(completed_trials, elapsed_secs):
    total_trials = len(experiments.experiment_list) * experiments.NUM_TRIALS
    percent_done = 100 * completed_trials // total_trials
    print(f'Completed {completed_trials} of '
          f'{total_trials} trials ({percent_done}%) | '
          f'Elapsed: {format_time(elapsed_secs)} | '
          f'Remaining: {format_time(estimate_remaining_time())}')


def estimate_remaining_time():
    # If there's no historical runtime data, assume a trial will take 20
    # minutes to run, which was typical in the dev environment.
    trial_duration_estimate = 20 * 60
    result = 0
    # Look at the experiments with the most completed trials first, since they
    # may have historical runtime data from this machine.
    experiment_list = sorted(
        experiments.experiment_list,
        key=lambda experiment: experiment.trial,
        reverse=True)
    for experiment in experiment_list:
        # If the current experiment has historical data, use that to estimate
        # how long it takes to run the remaining trials of this experiment.
        if experiment.average_trial_duration:
            # If some experiments haven't started yet but others have, then use
            # the true runtime from one experiment as an estimate for any
            # unstarted experiments.
            trial_duration_estimate = experiment.average_trial_duration
        result += trial_duration_estimate * experiment.remaining_trials
    return result


def run_experiments():
    # Handle Ctrl+C gracefully.
    signal.signal(signal.SIGINT, handle_sigint)

    # A single Simulator instance is reused for all trials and experiments.
    simulator = kernel.Simulator(
        experiments.NUM_SPECIES,
        experiments.NUM_TRIALS,
        experiments.NUM_ORGANISMS)

    # Review the experiment list and print a summary of the work to do.
    total_experiments = 0
    completed_trials = 0
    elapsed_secs = 0
    for experiment in experiments.experiment_list:
        total_experiments += 1
        completed_trials += experiment.trial + 1
        elapsed_secs += experiment.elapsed
    print(f'Running {len(experiments.experiment_list)} experiments, '
          f'{experiments.NUM_TRIALS} trials each.')
    print_status_summary(completed_trials, elapsed_secs)

    # Keep running trials until they're all done (or the user aborts). Always
    # pick the experiment with the fewest trials run next, so that no
    # experiment gets ahead of the others and partial results come in faster.
    while not user_requested_abort:
        experiment = min(
            experiments.experiment_list,
            key=lambda experiment: experiment.trial)
        if experiment.has_finished():
            print('All experiments completed.')
            break

        # Run the experiment and print an updated status summary.
        print(f'Running experiment {experiment.name}, '
              f'trial {experiment.trial + 1}')
        experiment.run_trial(simulator)
        completed_trials += 1
        elapsed_secs += experiment.elapsed
        print_status_summary(completed_trials, elapsed_secs)

        # Summarizing experiment data uses matplotlib which is prone to memory
        # leaks and isn't good for a long-running process like this one, so
        # spawn a separate process and don't wait for it to finish (note, the
        # Python GIL means these may not actually run simultaneously).
        subprocess.Popen(['python3', 'summarize_results.py'])


if __name__ == '__main__':
    run_experiments()
